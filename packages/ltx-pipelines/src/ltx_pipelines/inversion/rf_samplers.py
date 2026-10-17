from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent, state_with_conditionings


def _to_velocity(sample: torch.Tensor, denoised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if torch.any(sigma == 0):
        raise ValueError("Cannot convert denoised output to velocity at sigma=0.")
    while sigma.dim() < sample.dim():
        sigma = sigma.view(*sigma.shape, 1)
    return ((sample.to(torch.float32) - denoised.to(torch.float32)) / sigma.to(torch.float32)).to(sample.dtype)


def _trend_values(num_steps: int, start_step: int, end_step: int, value: float, trend: str, device: torch.device) -> torch.Tensor:
    values = torch.zeros(num_steps, dtype=torch.float32, device=device)
    if value == 0.0 or num_steps <= 0:
        return values
    start = max(0, min(start_step, num_steps))
    end = max(start, min(end_step, num_steps))
    if start == end:
        return values
    length = end - start
    if trend == "constant":
        window = torch.full((length,), value, dtype=torch.float32, device=device)
    elif trend == "linear_decrease":
        window = torch.linspace(value, 0.0, length, dtype=torch.float32, device=device)
    elif trend == "linear_increase":
        window = torch.linspace(0.0, value, length, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported trend: {trend}")
    values[start:end] = window
    return values


def _safe_divisor(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


@dataclass(frozen=True)
class RFSamplerResult:
    inverted_latent: torch.Tensor
    reconstructed_latent: torch.Tensor
    timesteps: torch.Tensor


class VelocityPredictor:
    def __call__(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        modality = modality_from_latent_state(state, condition.context, sigma)
        denoised, _ = transformer(
            video=modality,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(modality.latent.shape[0]),
        )
        if denoised is None:
            raise RuntimeError("Missing denoised video output from transformer.")
        denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        return _to_velocity(state.latent, denoised, sigma)


class ContextFlowRFSolver2Sampler:
    def __init__(
        self,
        *,
        video_tools: VideoLatentTools,
        eta: float = 0.8,
        eta_start_step: int = 0,
        eta_end_step: int = 15,
        eta_trend: str = "constant",
        use_reverse_pullback: bool = False,
        post_step_clamp: bool = False,
        first_order: bool = False,
        inversion_first_order: bool | None = None,
        reconstruction_first_order: bool | None = None,
        step_parameterization: str = "velocity",
        step_compute_dtype: torch.dtype = torch.float32,
        predictor: VelocityPredictor | None = None,
    ) -> None:
        self.video_tools = video_tools
        self.eta = eta
        self.eta_start_step = eta_start_step
        self.eta_end_step = eta_end_step
        self.eta_trend = eta_trend
        self.use_reverse_pullback = use_reverse_pullback
        self.post_step_clamp = post_step_clamp
        self.inversion_first_order = first_order if inversion_first_order is None else inversion_first_order
        self.reconstruction_first_order = first_order if reconstruction_first_order is None else reconstruction_first_order
        self.step_parameterization = step_parameterization
        self.step_compute_dtype = step_compute_dtype
        self.predictor = predictor or VelocityPredictor()

    def _predict_denoised(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        modality = modality_from_latent_state(state, condition.context, sigma)
        denoised, _ = transformer(
            video=modality,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(modality.latent.shape[0]),
        )
        if denoised is None:
            raise RuntimeError("Missing denoised video output during ContextFlow RF-Solver reconstruction.")
        return post_process_latent(denoised, state.denoise_mask, state.clean_latent)

    def _blend_with_pullback(
        self,
        *,
        velocity: torch.Tensor,
        latent: torch.Tensor,
        clean_source_latent: torch.Tensor,
        sigma: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        # ContextFlow reconstruction steps from high sigma to low sigma with a
        # negative dt, so the clean-pull velocity is the sign-flipped official
        # displacement: latent + ((latent - clean) / sigma) * (-sigma) = clean.
        pullback = (latent.to(torch.float32) - clean_source_latent.to(torch.float32)) / sigma.clamp_min(1e-6).to(torch.float32)
        return (velocity.to(torch.float32) + eta * (pullback - velocity.to(torch.float32))).to(velocity.dtype)

    def sample(
        self,
        *,
        transformer: X0Model,
        source_latent: torch.Tensor,
        condition: BranchCondition,
        timesteps: torch.Tensor,
    ) -> RFSamplerResult:
        # Keep the default baseline on the existing ContextFlow path. Only the
        # pullback ablation takes the custom branch below.
        from ltx_pipelines.inversion.attention.controller import NoOpAttentionController
        from ltx_pipelines.inversion.inversion import RectifiedFlowInverter
        from ltx_pipelines.inversion.trajectory import DualTrajectorySampler

        inversion = RectifiedFlowInverter(
            video_tools=self.video_tools,
            stepper=None,
            post_step_clamp=self.post_step_clamp,
            first_order=self.inversion_first_order,
            step_parameterization=self.step_parameterization,
            step_compute_dtype=self.step_compute_dtype,
        ).invert(
            transformer=transformer,
            source_latent=source_latent,
            source_condition=condition,
            timesteps=timesteps,
        )
        if not self.use_reverse_pullback:
            reconstructed_latent, _, _ = DualTrajectorySampler(
                video_tools=self.video_tools,
                stepper=None,
                post_step_clamp=self.post_step_clamp,
                first_order=self.reconstruction_first_order,
                step_parameterization=self.step_parameterization,
                step_compute_dtype=self.step_compute_dtype,
            ).sample(
                transformer=transformer,
                initial_noise=inversion.initial_noise,
                source_condition=condition,
                target_condition=condition,
                timesteps=inversion.timesteps,
                attention_controller=NoOpAttentionController(),
            )
            return RFSamplerResult(
                inverted_latent=inversion.initial_noise,
                reconstructed_latent=reconstructed_latent,
                timesteps=inversion.timesteps,
            )

        eta_values = _trend_values(
            len(inversion.timesteps) - 1,
            self.eta_start_step,
            self.eta_end_step,
            self.eta,
            self.eta_trend,
            inversion.initial_noise.device,
        )
        clean_state = self.video_tools.create_initial_state(source_latent.device, source_latent.dtype, source_latent)
        clean_state = state_with_conditionings(clean_state, condition.conditionings, self.video_tools)
        clean_source_latent = clean_state.latent.clone()
        state = self.video_tools.create_initial_state(
            inversion.initial_noise.device,
            inversion.initial_noise.dtype,
            inversion.initial_noise.clone(),
        )
        state = state_with_conditionings(state, condition.conditionings, self.video_tools)
        for step_index in range(len(inversion.timesteps) - 1):
            sigma = inversion.timesteps[step_index]
            sigma_next = inversion.timesteps[step_index + 1]
            midpoint_sigma = (sigma + sigma_next) * 0.5
            denoised = self._predict_denoised(transformer=transformer, state=state, condition=condition, sigma=sigma)
            velocity = _to_velocity(state.latent, denoised, sigma)
            velocity = self._blend_with_pullback(
                velocity=velocity,
                latent=state.latent,
                clean_source_latent=clean_source_latent,
                sigma=sigma,
                eta=eta_values[step_index],
            )
            midpoint_latent = state.latent.to(torch.float32) + velocity.to(torch.float32) * (sigma_next - sigma).to(torch.float32) * 0.5
            midpoint_state = LatentState(
                latent=midpoint_latent.to(state.latent.dtype),
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )
            midpoint_denoised = self._predict_denoised(
                transformer=transformer,
                state=midpoint_state,
                condition=condition,
                sigma=midpoint_sigma,
            )
            midpoint_velocity = (midpoint_state.latent.to(torch.float32) - midpoint_denoised.to(torch.float32)) / midpoint_sigma.to(torch.float32)
            midpoint_velocity = self._blend_with_pullback(
                velocity=midpoint_velocity.to(state.latent.dtype),
                latent=midpoint_state.latent,
                clean_source_latent=clean_source_latent,
                sigma=midpoint_sigma,
                eta=eta_values[step_index],
            )
            next_latent = state.latent.to(torch.float32) + midpoint_velocity.to(torch.float32) * (sigma_next - sigma).to(torch.float32)
            if self.post_step_clamp:
                next_latent = post_process_latent(next_latent.to(state.latent.dtype), state.denoise_mask, state.clean_latent)
            else:
                next_latent = next_latent.to(state.latent.dtype)
            state = LatentState(
                latent=next_latent,
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )
        reconstructed_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(state))
        return RFSamplerResult(
            inverted_latent=inversion.initial_noise,
            reconstructed_latent=reconstructed_state.latent,
            timesteps=inversion.timesteps,
        )


class LTXRFEditSampler:
    """RF-Edit-style RF-Solver-2 trajectory ablation.

    This mirrors the explicit second-order ODE update in LTX's
    ``rf_edit_sampler_nodes.py`` while using the local ``X0Model`` via
    ``VelocityPredictor``. It intentionally does not add RF-Inversion's
    gamma/eta pull terms.
    """

    def __init__(
        self,
        *,
        video_tools: VideoLatentTools,
        post_step_clamp: bool = False,
        predictor: VelocityPredictor | None = None,
    ) -> None:
        self.video_tools = video_tools
        self.post_step_clamp = post_step_clamp
        self.predictor = predictor or VelocityPredictor()

    def _predict(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(transformer=transformer, state=state, condition=condition, sigma=sigma)

    def _post_step_latent(self, state: LatentState, latent: torch.Tensor) -> torch.Tensor:
        if not self.post_step_clamp:
            return latent.to(state.latent.dtype)
        return post_process_latent(latent.to(state.latent.dtype), state.denoise_mask, state.clean_latent)

    def _solver2_step(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> LatentState:
        pred = self._predict(transformer=transformer, state=state, condition=condition, sigma=sigma).to(torch.float32)
        dt = (sigma_next - sigma).to(torch.float32)
        midpoint_latent = state.latent.to(torch.float32) + 0.5 * dt * pred
        midpoint_state = replace(state, latent=midpoint_latent.to(state.latent.dtype))
        sigma_mid = sigma + 0.5 * dt
        pred_mid = self._predict(
            transformer=transformer,
            state=midpoint_state,
            condition=condition,
            sigma=sigma_mid,
        ).to(torch.float32)
        first_order = (pred_mid - pred) / _safe_divisor(0.5 * dt)
        next_latent = state.latent.to(torch.float32) + dt * pred + 0.5 * dt * dt * first_order
        return replace(state, latent=self._post_step_latent(state, next_latent))

    def _conditioned_state(
        self,
        *,
        latent: torch.Tensor,
        condition: BranchCondition,
    ) -> LatentState:
        state = self.video_tools.create_initial_state(latent.device, latent.dtype, latent)
        return state_with_conditionings(state, condition.conditionings, self.video_tools)

    def sample(
        self,
        *,
        transformer: X0Model,
        source_latent: torch.Tensor,
        condition: BranchCondition,
        timesteps: torch.Tensor,
    ) -> RFSamplerResult:
        # ``timesteps`` is the usual denoising schedule: high sigma -> low sigma.
        # The forward/inversion pass therefore flips it to integrate clean -> noise.
        state = self._conditioned_state(latent=source_latent, condition=condition)
        forward_sigmas = torch.flip(timesteps[:-1], dims=[0]).contiguous()
        for step_index in range(len(forward_sigmas) - 1):
            state = self._solver2_step(
                transformer=transformer,
                state=state,
                condition=condition,
                sigma=forward_sigmas[step_index],
                sigma_next=forward_sigmas[step_index + 1],
            )
        inverted_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(state))
        inverted_latent = inverted_state.latent

        reconstruction_state = self._conditioned_state(latent=inverted_latent.clone(), condition=condition)
        for step_index in range(len(timesteps) - 1):
            reconstruction_state = self._solver2_step(
                transformer=transformer,
                state=reconstruction_state,
                condition=condition,
                sigma=timesteps[step_index],
                sigma_next=timesteps[step_index + 1],
            )
        reconstruction_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(reconstruction_state))
        return RFSamplerResult(
            inverted_latent=inverted_latent,
            reconstructed_latent=reconstruction_state.latent,
            timesteps=timesteps,
        )


class LTXRFInversionSampler:
    def __init__(
        self,
        *,
        video_tools: VideoLatentTools,
        gamma: float = 0.5,
        eta: float = 0.8,
        gamma_start_step: int = 0,
        gamma_end_step: int = 5,
        eta_start_step: int = 0,
        eta_end_step: int = 15,
        gamma_trend: str = "linear_decrease",
        eta_trend: str = "constant",
        order: str = "first",
        seed: int = 0,
        use_reverse_pullback: bool = True,
        predictor: VelocityPredictor | None = None,
    ) -> None:
        if order not in {"first", "second"}:
            raise ValueError(f"order must be 'first' or 'second', got {order!r}")
        self.video_tools = video_tools
        self.gamma = gamma
        self.eta = eta
        self.gamma_start_step = gamma_start_step
        self.gamma_end_step = gamma_end_step
        self.eta_start_step = eta_start_step
        self.eta_end_step = eta_end_step
        self.gamma_trend = gamma_trend
        self.eta_trend = eta_trend
        self.order = order
        self.seed = seed
        self.use_reverse_pullback = use_reverse_pullback
        self.predictor = predictor or VelocityPredictor()

    def _predict(self, *, transformer: X0Model, state: LatentState, condition: BranchCondition, sigma: torch.Tensor) -> torch.Tensor:
        return self.predictor(transformer=transformer, state=state, condition=condition, sigma=sigma)

    def _forward_step(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        target_noise: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        gamma: torch.Tensor,
    ) -> LatentState:
        # Official forward RF-Inversion uses (y1 - Y) / (1 - t_i).
        # In this LTX adapter, the model is called with sigma directly, so t_i maps to sigma.
        conditional_vector_field = (target_noise.to(torch.float32) - state.latent.to(torch.float32)) / (1.0 - sigma).clamp_min(1e-6).to(torch.float32)
        pred = self._predict(transformer=transformer, state=state, condition=condition, sigma=sigma).to(torch.float32)
        dt = (sigma_next - sigma).to(torch.float32)
        if self.order == "second":
            pred_order_for_mid = pred + gamma * (conditional_vector_field - pred)
            midpoint_latent = state.latent.to(torch.float32) + dt * 0.5 * pred
            midpoint_state = LatentState(
                latent=midpoint_latent.to(state.latent.dtype),
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )
            sigma_mid = sigma + dt * 0.5
            # RF-Inversion follows the direct vector-field midpoint call: v(x_mid, sigma_mid).
            pred_mid = self._predict(transformer=transformer, state=midpoint_state, condition=condition, sigma=sigma_mid).to(torch.float32)
            first_order = (pred_mid - pred) / _safe_divisor(dt * 0.5)
            next_latent = state.latent.to(torch.float32) + dt * pred_order_for_mid + 0.5 * dt * dt * first_order
        else:
            pred = pred + gamma * (conditional_vector_field - pred)
            next_latent = state.latent.to(torch.float32) + pred * dt
        next_latent = post_process_latent(next_latent.to(state.latent.dtype), state.denoise_mask, state.clean_latent)
        return LatentState(
            latent=next_latent,
            denoise_mask=state.denoise_mask,
            positions=state.positions,
            clean_latent=state.clean_latent,
            attention_mask=state.attention_mask,
        )

    def _reverse_step(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        source_state_latent: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        eta: torch.Tensor,
    ) -> LatentState:
        # Official reverse RF-Inversion uses t_i = 1 - timestep(sigma), then (y0 - X) / (1 - t_i).
        # With local timestep(sigma) == sigma, the denominator is sigma.
        conditional_vector_field = (source_state_latent.to(torch.float32) - state.latent.to(torch.float32)) / sigma.clamp_min(1e-6).to(torch.float32)
        pred = self._predict(transformer=transformer, state=state, condition=condition, sigma=sigma).to(torch.float32)
        dt = (sigma - sigma_prev).to(torch.float32)
        if self.order == "second":
            midpoint_latent = state.latent.to(torch.float32) + (sigma_prev - sigma).to(torch.float32) * 0.5 * pred
            midpoint_state = LatentState(
                latent=midpoint_latent.to(state.latent.dtype),
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )
            sigma_mid = sigma + (sigma_prev - sigma) * 0.5
            # RF-Inversion follows the direct vector-field midpoint call: v(x_mid, sigma_mid).
            pred_mid = self._predict(transformer=transformer, state=midpoint_state, condition=condition, sigma=sigma_mid).to(torch.float32)
            first_order = (pred_mid - pred) / _safe_divisor((sigma_prev - sigma).to(torch.float32) * 0.5)
            if self.use_reverse_pullback:
                pred = -pred + eta * (conditional_vector_field + pred)
                first_order = -first_order + eta * (conditional_vector_field + first_order)
            else:
                pred = -pred
                first_order = -first_order
            next_latent = state.latent.to(torch.float32) + dt * pred + 0.5 * dt * dt * first_order
        else:
            if self.use_reverse_pullback:
                controlled_vector_field = -pred + eta * (conditional_vector_field + pred)
            else:
                controlled_vector_field = -pred
            next_latent = state.latent.to(torch.float32) + controlled_vector_field * dt
        next_latent = post_process_latent(next_latent.to(state.latent.dtype), state.denoise_mask, state.clean_latent)
        return LatentState(
            latent=next_latent,
            denoise_mask=state.denoise_mask,
            positions=state.positions,
            clean_latent=state.clean_latent,
            attention_mask=state.attention_mask,
        )

    def sample(
        self,
        *,
        transformer: X0Model,
        source_latent: torch.Tensor,
        condition: BranchCondition,
        timesteps: torch.Tensor,
    ) -> RFSamplerResult:
        num_steps = len(timesteps) - 1
        gamma_values = _trend_values(num_steps, self.gamma_start_step, self.gamma_end_step, self.gamma, self.gamma_trend, source_latent.device)
        eta_values = _trend_values(num_steps, self.eta_start_step, self.eta_end_step, self.eta, self.eta_trend, source_latent.device)

        state = self.video_tools.create_initial_state(source_latent.device, source_latent.dtype, source_latent)
        state = state_with_conditionings(state, condition.conditionings, self.video_tools)
        clean_source_latent = state.latent.clone()
        generator = torch.Generator(device=state.latent.device).manual_seed(self.seed)
        target_noise = torch.randn(state.latent.shape, generator=generator, device=state.latent.device, dtype=state.latent.dtype)

        # Official forward RF-Inversion runs clean -> noise, so use the ascending nonzero schedule.
        # forward_sigmas = torch.flip(timesteps[:-1], dims=[0]).contiguous()
        forward_sigmas = torch.flip(timesteps, dims=[0]).contiguous()
        for step_index in range(len(forward_sigmas) - 1):
            state = self._forward_step(
                transformer=transformer,
                state=state,
                condition=condition,
                target_noise=target_noise,
                sigma=forward_sigmas[step_index],
                sigma_next=forward_sigmas[step_index + 1],
                gamma=gamma_values[step_index],
            )
        inverted_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(state))
        inverted_latent = inverted_state.latent

        reconstruction_state = self.video_tools.create_initial_state(inverted_latent.device, inverted_latent.dtype, inverted_latent.clone())
        reconstruction_state = state_with_conditionings(reconstruction_state, condition.conditionings, self.video_tools)
        for step_index in range(len(timesteps) - 1):
            reconstruction_state = self._reverse_step(
                transformer=transformer,
                state=reconstruction_state,
                condition=condition,
                source_state_latent=clean_source_latent,
                sigma=timesteps[step_index],
                sigma_prev=timesteps[step_index + 1],
                eta=eta_values[step_index],
            )
        reconstruction_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(reconstruction_state))
        return RFSamplerResult(inverted_latent=inverted_latent, reconstructed_latent=reconstruction_state.latent, timesteps=timesteps)
