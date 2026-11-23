"""ILMU PAYG token pricing table (MYR per 1M tokens, verified 2026-11-23)."""

from __future__ import annotations

from dataclasses import dataclass

from malay_tool_bench.models import TokenUsage


@dataclass(frozen=True, slots=True)
class ModelRate:
    """Rates in MYR per 1,000,000 tokens."""

    input_myr_per_m: float
    cached_input_myr_per_m: float
    output_myr_per_m: float


ILMU_RATES_MYR: dict[str, ModelRate] = {
    "ilmu-mini-v3.3": ModelRate(0.20, 0.02, 1.20),
    "ilmu-glm-5.3-flash": ModelRate(0.55, 0.10, 1.85),
    "ilmu-v3.1": ModelRate(4.00, 0.40, 16.00),
    "ilmu-glm-5.3": ModelRate(5.15, 0.95, 16.05),
    "ilmu-vision-v1.3": ModelRate(1.60, 0.16, 4.80),
}


def has_known_price(model_id: str) -> bool:
    """Return True if the model has a known MYR rate in the pricing table."""
    return model_id in ILMU_RATES_MYR


def compute_cost_myr(model_id: str, usage: TokenUsage) -> float:
    """Compute exact MYR cost from API token usage, honoring cached input discounts."""
    rate = ILMU_RATES_MYR.get(model_id)
    if rate is None:
        return 0.0
    cached = min(max(0, usage.cached_tokens), max(0, usage.prompt_tokens))
    uncached = max(0, usage.prompt_tokens - cached)
    output = max(0, usage.completion_tokens)
    return (
        uncached * rate.input_myr_per_m
        + cached * rate.cached_input_myr_per_m
        + output * rate.output_myr_per_m
    ) / 1_000_000.0
