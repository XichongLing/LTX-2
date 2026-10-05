from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SourceStrengthScheduleKind(str, Enum):
    CONSTANT = "constant"
    VALUES = "values"
    SIGMA_RAMP = "sigma-ramp"


class SourceStrengthRouting(str, Enum):
    SYMMETRIC = "symmetric"
    TARGET_QUERIES_ONLY = "target-queries-only"


@dataclass(frozen=True)
class SourceStrengthEvaluation:
    evaluation_index: int
    nominal_step_index: int
    sigma: float
    next_sigma: float | None
    progress: float
    g_source: float
    routing: SourceStrengthRouting
    target_token_count: int
    source_token_ranges: tuple[tuple[int, int], ...]
    composed_existing_mask: bool
    num_evaluations: int | None
    source_range_strengths: tuple[float, ...] = ()


class SourceStrengthSchedule:
    kind: SourceStrengthScheduleKind

    def strength(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        raise NotImplementedError


def _validate_strength(value: float, label: str = "source strength") -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class ConstantSourceStrengthSchedule(SourceStrengthSchedule):
    value: float = 1.0
    kind: SourceStrengthScheduleKind = SourceStrengthScheduleKind.CONSTANT

    def __post_init__(self) -> None:
        _validate_strength(self.value)

    def strength(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (sigma, progress, evaluation_index, num_evaluations)
        return float(self.value)


@dataclass(frozen=True)
class ValuesSourceStrengthSchedule(SourceStrengthSchedule):
    values: tuple[float, ...]
    kind: SourceStrengthScheduleKind = SourceStrengthScheduleKind.VALUES

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("source strength values must not be empty")
        for i, value in enumerate(self.values):
            _validate_strength(value, f"source strength value {i}")

    def strength(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (sigma, progress)
        if evaluation_index >= len(self.values):
            raise ValueError(
                "source strength values length does not cover evaluation "
                f"{evaluation_index}; got {len(self.values)} values"
            )
        if num_evaluations is not None and num_evaluations != len(self.values):
            raise ValueError(
                "source strength values length must match the number of transformer evaluations: "
                f"got {len(self.values)} values for {num_evaluations} evaluations"
            )
        return float(self.values[evaluation_index])


@dataclass(frozen=True)
class SigmaRampSourceStrengthSchedule(SourceStrengthSchedule):
    early_strength: float = 1.0
    late_strength: float = 0.3
    fade_start_sigma: float = 0.725
    fade_end_sigma: float = 0.421875
    kind: SourceStrengthScheduleKind = SourceStrengthScheduleKind.SIGMA_RAMP

    def __post_init__(self) -> None:
        _validate_strength(self.early_strength, "early source strength")
        _validate_strength(self.late_strength, "late source strength")
        for label, value in (
            ("source strength fade start sigma", self.fade_start_sigma),
            ("source strength fade end sigma", self.fade_end_sigma),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite, got {value}")
        if self.fade_start_sigma < self.fade_end_sigma:
            raise ValueError(
                "source strength fade start sigma must be >= fade end sigma for descending RF sigmas"
            )

    def strength(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (progress, evaluation_index, num_evaluations)
        sigma = float(sigma)
        if sigma >= self.fade_start_sigma:
            return float(self.early_strength)
        if sigma <= self.fade_end_sigma:
            return float(self.late_strength)
        span = self.fade_start_sigma - self.fade_end_sigma
        alpha = (self.fade_start_sigma - sigma) / span
        return float((1.0 - alpha) * self.early_strength + alpha * self.late_strength)


def parse_source_strength_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return ValuesSourceStrengthSchedule(values).values


def build_source_strength_schedule(
    kind: SourceStrengthScheduleKind | str,
    *,
    strength: float = 1.0,
    values: tuple[float, ...] | None = None,
    early_strength: float = 1.0,
    late_strength: float = 0.3,
    fade_start_sigma: float = 0.725,
    fade_end_sigma: float = 0.421875,
) -> SourceStrengthSchedule:
    kind = SourceStrengthScheduleKind(kind)
    if kind == SourceStrengthScheduleKind.CONSTANT:
        return ConstantSourceStrengthSchedule(strength)
    if kind == SourceStrengthScheduleKind.VALUES:
        if values is None:
            raise ValueError("source strength values are required when schedule kind is 'values'")
        return ValuesSourceStrengthSchedule(values)
    if kind == SourceStrengthScheduleKind.SIGMA_RAMP:
        return SigmaRampSourceStrengthSchedule(
            early_strength=early_strength,
            late_strength=late_strength,
            fade_start_sigma=fade_start_sigma,
            fade_end_sigma=fade_end_sigma,
        )
    raise ValueError(f"Unsupported source strength schedule kind: {kind}")


DEFAULT_FIRST_FRAME_ATTENTION_MAX = 16.0


class FirstFrameAttentionScheduleKind(str, Enum):
    CONSTANT = "constant"
    VALUES = "values"
    SIGMA_RAMP = "sigma-ramp"


@dataclass(frozen=True)
class FirstFrameAttentionEvaluation:
    evaluation_index: int
    nominal_step_index: int
    sigma: float
    next_sigma: float | None
    progress: float
    h_first_frame: float
    target_token_count: int
    frame0_token_range: tuple[int, int]
    later_target_token_range: tuple[int, int]
    composed_existing_mask: bool
    num_evaluations: int | None


class FirstFrameAttentionSchedule:
    kind: FirstFrameAttentionScheduleKind

    def multiplier(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        raise NotImplementedError


def _validate_multiplier(
    value: float,
    label: str = "first-frame attention multiplier",
    *,
    maximum: float = DEFAULT_FIRST_FRAME_ATTENTION_MAX,
) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    if not 0.0 <= value <= maximum:
        raise ValueError(f"{label} must be in [0, {maximum:g}], got {value}")
    return value


@dataclass(frozen=True)
class ConstantFirstFrameAttentionSchedule(FirstFrameAttentionSchedule):
    value: float = 1.0
    maximum: float = DEFAULT_FIRST_FRAME_ATTENTION_MAX
    kind: FirstFrameAttentionScheduleKind = FirstFrameAttentionScheduleKind.CONSTANT

    def __post_init__(self) -> None:
        _validate_multiplier(self.value, maximum=self.maximum)

    def multiplier(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (sigma, progress, evaluation_index, num_evaluations)
        return float(self.value)


@dataclass(frozen=True)
class ValuesFirstFrameAttentionSchedule(FirstFrameAttentionSchedule):
    values: tuple[float, ...]
    maximum: float = DEFAULT_FIRST_FRAME_ATTENTION_MAX
    kind: FirstFrameAttentionScheduleKind = FirstFrameAttentionScheduleKind.VALUES

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("first-frame attention values must not be empty")
        for i, value in enumerate(self.values):
            _validate_multiplier(value, f"first-frame attention value {i}", maximum=self.maximum)

    def multiplier(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (sigma, progress)
        if evaluation_index >= len(self.values):
            raise ValueError(
                "first-frame attention values length does not cover evaluation "
                f"{evaluation_index}; got {len(self.values)} values"
            )
        if num_evaluations is not None and num_evaluations != len(self.values):
            raise ValueError(
                "first-frame attention values length must match the number of transformer evaluations: "
                f"got {len(self.values)} values for {num_evaluations} evaluations"
            )
        return float(self.values[evaluation_index])


@dataclass(frozen=True)
class SigmaRampFirstFrameAttentionSchedule(FirstFrameAttentionSchedule):
    early_multiplier: float = 1.0
    late_multiplier: float = 4.0
    fade_start_sigma: float = 0.975
    fade_end_sigma: float = 0.421875
    maximum: float = DEFAULT_FIRST_FRAME_ATTENTION_MAX
    kind: FirstFrameAttentionScheduleKind = FirstFrameAttentionScheduleKind.SIGMA_RAMP

    def __post_init__(self) -> None:
        _validate_multiplier(self.early_multiplier, "early first-frame attention multiplier", maximum=self.maximum)
        _validate_multiplier(self.late_multiplier, "late first-frame attention multiplier", maximum=self.maximum)
        for label, value in (
            ("first-frame attention fade start sigma", self.fade_start_sigma),
            ("first-frame attention fade end sigma", self.fade_end_sigma),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite, got {value}")
        if self.fade_start_sigma < self.fade_end_sigma:
            raise ValueError(
                "first-frame attention fade start sigma must be >= fade end sigma for descending RF sigmas"
            )

    def multiplier(
        self,
        *,
        sigma: float,
        progress: float,
        evaluation_index: int,
        num_evaluations: int | None,
    ) -> float:
        _ = (progress, evaluation_index, num_evaluations)
        sigma = float(sigma)
        if sigma >= self.fade_start_sigma:
            return float(self.early_multiplier)
        if sigma <= self.fade_end_sigma:
            return float(self.late_multiplier)
        span = self.fade_start_sigma - self.fade_end_sigma
        alpha = (self.fade_start_sigma - sigma) / span
        return float((1.0 - alpha) * self.early_multiplier + alpha * self.late_multiplier)


def parse_first_frame_attention_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return ValuesFirstFrameAttentionSchedule(values).values


def build_first_frame_attention_schedule(
    kind: FirstFrameAttentionScheduleKind | str,
    *,
    multiplier: float = 1.0,
    values: tuple[float, ...] | None = None,
    early_multiplier: float = 1.0,
    late_multiplier: float = 4.0,
    fade_start_sigma: float = 0.975,
    fade_end_sigma: float = 0.421875,
    maximum: float = DEFAULT_FIRST_FRAME_ATTENTION_MAX,
) -> FirstFrameAttentionSchedule:
    kind = FirstFrameAttentionScheduleKind(kind)
    if kind == FirstFrameAttentionScheduleKind.CONSTANT:
        return ConstantFirstFrameAttentionSchedule(multiplier, maximum=maximum)
    if kind == FirstFrameAttentionScheduleKind.VALUES:
        if values is None:
            raise ValueError("first-frame attention values are required when schedule kind is 'values'")
        return ValuesFirstFrameAttentionSchedule(values, maximum=maximum)
    if kind == FirstFrameAttentionScheduleKind.SIGMA_RAMP:
        return SigmaRampFirstFrameAttentionSchedule(
            early_multiplier=early_multiplier,
            late_multiplier=late_multiplier,
            fade_start_sigma=fade_start_sigma,
            fade_end_sigma=fade_end_sigma,
            maximum=maximum,
        )
    raise ValueError(f"Unsupported first-frame attention schedule kind: {kind}")
