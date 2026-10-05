"""Flat denoiser classes — transformer received at call time, not stored.
Three implementations of the :class:`~ltx_pipelines.utils.types.Denoiser` protocol:
* :class:`SimpleDenoiser` — single transformer call, no guidance.
* :class:`GuidedDenoiser` — static guiders, handles CFG + STG + isolated modality.
* :class:`FactoryGuidedDenoiser` — resolves guiders per-step from sigma.
``GuidedDenoiser`` and ``FactoryGuidedDenoiser`` share the core multi-pass
logic via the module-level :func:`_guided_denoise` function, which batches
all guidance passes into a single transformer call.
"""

from collections.abc import Callable
import math
from dataclasses import replace

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.conditioning_schedules import (
    FirstFrameAttentionEvaluation,
    FirstFrameAttentionSchedule,
    SourceStrengthEvaluation,
    SourceStrengthRouting,
    SourceStrengthSchedule,
)
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.types import DenoisedLatentResult

_POSITIVE_ONLY_GUIDER = MultiModalGuider(
    params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0),
)
"""Guider that only runs the conditioned pass and returns cond unchanged."""


def _ensure_guider(guider: MultiModalGuider | None) -> MultiModalGuider:
    """Return the guider as-is, or a positive-only guider for absent modalities."""
    return guider if guider is not None else _POSITIVE_ONLY_GUIDER


def _repeat_state(state: LatentState, n: int) -> LatentState:
    """Repeat a ``LatentState`` *n* times along the batch dimension.
    ``(B, ...) → (n*B, ...)`` by tiling the whole tensor n times, so the
    ordering is ``[item0, item1, ..., item0, item1, ...]`` — matching
    ``torch.cat`` of n per-pass contexts.
    """

    def _repeat(t: torch.Tensor) -> torch.Tensor:
        repeats = [1] * t.dim()
        repeats[0] = n
        return t.repeat(repeats)

    return LatentState(
        latent=_repeat(state.latent),
        denoise_mask=_repeat(state.denoise_mask),
        positions=_repeat(state.positions),
        clean_latent=_repeat(state.clean_latent),
        attention_mask=_repeat(state.attention_mask) if state.attention_mask is not None else None,
    )


def _guided_denoise(  # noqa: PLR0913,PLR0915
    transformer: X0Model,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    sigma: torch.Tensor,
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor | None,
    a_context: torch.Tensor | None,
    *,
    last_denoised_video: torch.Tensor | None,
    last_denoised_audio: torch.Tensor | None,
    step_index: int,
    force_uncond_pass: bool = False,
) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
    """Core guided denoising — batches all guidance passes into one transformer call.
    Collects per-pass contexts first, then builds a single batched Modality
    per present modality via :func:`modality_from_latent_state`.  When wrapped
    with :class:`~ltx_core.batch_split.BatchSplitAdapter`, the transformer may
    split this batch into sequential chunks internally.
    Guiders must not be ``None``. For absent modalities, callers should pass
    :data:`_POSITIVE_ONLY_GUIDER` (via :func:`_ensure_guider`) so that only
    the conditioned pass runs and ``calculate()`` returns cond unchanged.
    """
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        video_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_video)
        audio_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)
        return video_result, audio_result

    if video_state is not None and v_context is None:
        raise ValueError("v_context is required when video_state is provided")
    if audio_state is not None and a_context is None:
        raise ValueError("a_context is required when audio_state is provided")
    # Define passes: (name, video_context, audio_context, perturbation_config).
    # Context is None for absent modalities — filtered out during collection.
    _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
    passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

    v_needs_neg = video_guider.do_unconditional_generation() or (force_uncond_pass and video_state is not None)
    a_needs_neg = audio_guider.do_unconditional_generation() or (force_uncond_pass and audio_state is not None)
    if v_needs_neg or a_needs_neg:
        if v_needs_neg and video_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        if a_needs_neg and audio_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations: list[Perturbation] = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
        )
    if stg_perturbations:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                        Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                    ]
                ),
            )
        )

    # Collect contexts, repeat states, and build batched modalities.
    pass_names = [name for name, _, _, _ in passes]
    ptb_configs = [ptb for _, _, _, ptb in passes]
    n = len(passes)

    orig_b = (video_state or audio_state).latent.shape[0]

    def _batched_sigma(state: LatentState) -> torch.Tensor:
        """Expand scalar sigma to (n * B,) matching the repeated state."""
        return sigma.expand(state.latent.shape[0] * n)

    batched_video = None
    if video_state is not None:
        v_context = torch.cat([vc for _, vc, _, _ in passes], dim=0)
        batched_video = modality_from_latent_state(
            _repeat_state(video_state, n),
            v_context,
            _batched_sigma(video_state),
            enabled=not v_skip,
        )

    batched_audio = None
    if audio_state is not None:
        a_context = torch.cat([ac for _, _, ac, _ in passes], dim=0)
        batched_audio = modality_from_latent_state(
            _repeat_state(audio_state, n),
            a_context,
            _batched_sigma(audio_state),
            enabled=not a_skip,
        )

    # Replicate each pass's PerturbationConfig to all `orig_b` samples it
    # carries, so `BatchedPerturbationConfig.mask_like` returns a per-sample
    # mask (length n*orig_b) instead of a per-pass mask (length n). Without
    # this expansion the mask is broadcast against a (n*orig_b, T, D) tensor
    # and the multiplication fails with a batch-dim mismatch whenever
    # `orig_b > 1` (e.g. multi-prompt benchmark panels).
    batched_ptb_configs = [ptb for ptb in ptb_configs for _ in range(orig_b)]

    all_v, all_a = transformer(
        video=batched_video, audio=batched_audio, perturbations=BatchedPerturbationConfig(batched_ptb_configs)
    )

    # Split results back and combine via guiders.
    splits_v = list(all_v.chunk(n)) if all_v is not None else [0.0] * n
    splits_a = list(all_a.chunk(n)) if all_a is not None else [0.0] * n
    r = dict(zip(pass_names, zip(splits_v, splits_a, strict=True), strict=True))

    cond_v, cond_a = r["cond"]
    uncond_v, uncond_a = r.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = r.get("ptb", (0.0, 0.0))
    mod_v, mod_a = r.get("mod", (0.0, 0.0))

    denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
    denoised_audio = last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
    return (
        DenoisedLatentResult.result_or_none(
            denoised=denoised_video, uncond=uncond_v, cond=cond_v, ptb=ptb_v, mod=mod_v
        ),
        DenoisedLatentResult.result_or_none(
            denoised=denoised_audio, uncond=uncond_a, cond=cond_a, ptb=ptb_a, mod=mod_a
        ),
    )


class SimpleDenoiser:
    """Single transformer call, no guidance.
    Passes ``None`` Modality for absent modalities.
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


class SourceStrengthScheduledDenoiser:
    """Scale scheduled video self-attention blocks immediately before each model call."""

    def __init__(
        self,
        inner: object,
        *,
        schedule: SourceStrengthSchedule,
        target_token_count: int,
        source_token_counts: tuple[int, ...],
        source_range_strengths: tuple[float, ...] | None = None,
        routing: SourceStrengthRouting = SourceStrengthRouting.SYMMETRIC,
        num_evaluations: int | None = None,
        sink: Callable[[SourceStrengthEvaluation], None] | None = None,
        first_frame_schedule: FirstFrameAttentionSchedule | None = None,
        first_frame_token_count: int | None = None,
        first_frame_sink: Callable[[FirstFrameAttentionEvaluation], None] | None = None,
    ) -> None:
        self.inner = inner
        self.schedule = schedule
        self.target_token_count = target_token_count
        self.source_token_counts = source_token_counts
        self.source_range_strengths = source_range_strengths or tuple(1.0 for _ in source_token_counts)
        if len(self.source_range_strengths) != len(source_token_counts):
            raise ValueError("source_range_strengths must match source_token_counts")
        if any(not math.isfinite(value) or value < 0.0 for value in self.source_range_strengths):
            raise ValueError("source_range_strengths must be finite and non-negative")
        self.routing = SourceStrengthRouting(routing)
        self.num_evaluations = num_evaluations
        self.sink = sink
        self.first_frame_schedule = first_frame_schedule
        self.first_frame_token_count = first_frame_token_count
        self.first_frame_sink = first_frame_sink
        self._evaluation_index = 0

    def _source_token_ranges(self, total_tokens: int) -> tuple[tuple[int, int], ...]:
        source_total = sum(self.source_token_counts)
        if source_total <= 0:
            return ()
        if total_tokens < self.target_token_count + source_total:
            raise ValueError(
                f"latent state has {total_tokens} tokens, which cannot contain "
                f"{self.target_token_count} target tokens and {source_total} source tokens"
            )
        start = total_tokens - source_total
        ranges = []
        for count in self.source_token_counts:
            stop = start + count
            ranges.append((start, stop))
            start = stop
        return tuple(ranges)

    def _frame0_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.first_frame_token_count is None:
            raise ValueError("first_frame_token_count is required when first_frame_schedule is set")
        if not 0 < self.first_frame_token_count <= self.target_token_count:
            raise ValueError(
                "first_frame_token_count must be positive and no larger than target_token_count: "
                f"got {self.first_frame_token_count} and {self.target_token_count}"
            )
        return (0, self.first_frame_token_count), (self.first_frame_token_count, self.target_token_count)

    def _attention_mask_for_edit(self, video_state: LatentState) -> tuple[torch.Tensor, bool]:
        total_tokens = video_state.latent.shape[1]
        if video_state.attention_mask is None:
            return (
                torch.ones(
                    (video_state.latent.shape[0], total_tokens, total_tokens),
                    device=video_state.latent.device,
                    dtype=video_state.latent.dtype,
                ),
                False,
            )
        return video_state.attention_mask.clone(), True

    def _scheduled_video_state(
        self,
        video_state: LatentState,
        *,
        g_source: float,
        h_first_frame: float,
    ) -> tuple[
        LatentState,
        bool,
        tuple[tuple[int, int], ...],
        tuple[int, int] | None,
        tuple[int, int] | None,
    ]:
        total_tokens = video_state.latent.shape[1]
        source_ranges = self._source_token_ranges(total_tokens)
        effective_strengths = tuple(g_source * value for value in self.source_range_strengths)
        needs_source_edit = bool(source_ranges) and any(value != 1.0 for value in effective_strengths)
        needs_frame0_edit = self.first_frame_schedule is not None and h_first_frame != 1.0
        frame0_range = None
        later_target_range = None
        if self.first_frame_schedule is not None:
            frame0_range, later_target_range = self._frame0_ranges()

        if not needs_source_edit and not needs_frame0_edit:
            return video_state, video_state.attention_mask is not None, source_ranges, frame0_range, later_target_range

        attention_mask, composed_existing_mask = self._attention_mask_for_edit(video_state)

        if needs_source_edit:
            for (start, stop), effective_strength in zip(source_ranges, effective_strengths, strict=True):
                attention_mask[:, : self.target_token_count, start:stop] *= effective_strength
                if self.routing == SourceStrengthRouting.SYMMETRIC:
                    attention_mask[:, start:stop, : self.target_token_count] *= effective_strength

        if needs_frame0_edit:
            assert frame0_range is not None
            assert later_target_range is not None
            frame0_start, frame0_stop = frame0_range
            later_start, later_stop = later_target_range
            attention_mask[:, later_start:later_stop, frame0_start:frame0_stop] *= h_first_frame

        return replace(video_state, attention_mask=attention_mask), composed_existing_mask, source_ranges, frame0_range, later_target_range

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        sigma = sigmas[step_index]
        next_sigma = sigmas[step_index + 1] if step_index + 1 < sigmas.numel() else None
        if self.num_evaluations is not None and self.num_evaluations > 1:
            progress = self._evaluation_index / (self.num_evaluations - 1)
        else:
            progress = 0.0
        sigma_float = float(sigma.detach().cpu())
        next_sigma_float = float(next_sigma.detach().cpu()) if next_sigma is not None else None
        g_source = self.schedule.strength(
            sigma=sigma_float,
            progress=progress,
            evaluation_index=self._evaluation_index,
            num_evaluations=self.num_evaluations,
        )
        h_first_frame = (
            self.first_frame_schedule.multiplier(
                sigma=sigma_float,
                progress=progress,
                evaluation_index=self._evaluation_index,
                num_evaluations=self.num_evaluations,
            )
            if self.first_frame_schedule is not None
            else 1.0
        )

        composed_existing_mask = False
        source_ranges: tuple[tuple[int, int], ...] = ()
        frame0_range: tuple[int, int] | None = None
        later_target_range: tuple[int, int] | None = None
        if video_state is not None:
            video_state, composed_existing_mask, source_ranges, frame0_range, later_target_range = (
                self._scheduled_video_state(video_state, g_source=g_source, h_first_frame=h_first_frame)
            )

        if self.sink is not None:
            self.sink(
                SourceStrengthEvaluation(
                    evaluation_index=self._evaluation_index,
                    nominal_step_index=step_index,
                    sigma=sigma_float,
                    next_sigma=next_sigma_float,
                    progress=progress,
                    g_source=g_source,
                    routing=self.routing,
                    target_token_count=self.target_token_count,
                    source_token_ranges=source_ranges,
                    composed_existing_mask=composed_existing_mask,
                    num_evaluations=self.num_evaluations,
                    source_range_strengths=tuple(
                        g_source * value for value in self.source_range_strengths
                    ),
                )
            )
        if self.first_frame_sink is not None and self.first_frame_schedule is not None:
            if frame0_range is None or later_target_range is None:
                frame0_range, later_target_range = self._frame0_ranges()
            self.first_frame_sink(
                FirstFrameAttentionEvaluation(
                    evaluation_index=self._evaluation_index,
                    nominal_step_index=step_index,
                    sigma=sigma_float,
                    next_sigma=next_sigma_float,
                    progress=progress,
                    h_first_frame=h_first_frame,
                    target_token_count=self.target_token_count,
                    frame0_token_range=frame0_range,
                    later_target_token_range=later_target_range,
                    composed_existing_mask=composed_existing_mask,
                    num_evaluations=self.num_evaluations,
                )
            )
        self._evaluation_index += 1
        return self.inner(transformer, video_state, audio_state, sigmas, step_index)


class GuidedDenoiser:
    """Static guiders — handles CFG + STG + isolated modality.
    Context/guider can be ``None`` for absent modalities (a positive-only
    guider is substituted at call time).
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider: MultiModalGuider | None = None,
        audio_guider: MultiModalGuider | None = None,
        force_uncond_pass: bool = False,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.force_uncond_pass = force_uncond_pass
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=_ensure_guider(self.video_guider),
            audio_guider=_ensure_guider(self.audio_guider),
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised
        self._last_denoised_audio = guided_denoise_result_a.denoised
        return guided_denoise_result_v, guided_denoise_result_a


class FactoryGuidedDenoiser:
    """Resolves guiders per-step from sigma, then delegates to shared guided logic."""

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider_factory: MultiModalGuiderFactory | None = None,
        audio_guider_factory: MultiModalGuiderFactory | None = None,
        force_uncond_pass: bool = False,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider_factory = video_guider_factory
        self.audio_guider_factory = audio_guider_factory
        self.force_uncond_pass = force_uncond_pass
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None
        self._sigma_vals_cached: list[float] | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if self._sigma_vals_cached is None:
            self._sigma_vals_cached = sigmas.detach().cpu().tolist()
        sigma_val = self._sigma_vals_cached[step_index]

        video_guider = _ensure_guider(
            self.video_guider_factory.build_from_sigma(sigma_val) if self.video_guider_factory else None
        )
        audio_guider = _ensure_guider(
            (self.audio_guider_factory or self.video_guider_factory).build_from_sigma(sigma_val)
            if self.video_guider_factory or self.audio_guider_factory
            else None
        )

        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=video_guider,
            audio_guider=audio_guider,
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised
        self._last_denoised_audio = guided_denoise_result_a.denoised
        return guided_denoise_result_v, guided_denoise_result_a
