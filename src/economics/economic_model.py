"""
End-to-End Net Economic Benefit & Latency Model.
Quantifies composite financial cost and latency:
Total Cost = (Input Tokens * Price) + (Output Tokens * Price) + Compression Compute Cost + (Failure Rate * Retry Cost)
Evaluates latency percentiles (p50, p95, p99) under load.
"""

import numpy as np
from typing import Any, Dict, List
from src.integrations.openai_client import MODEL_INPUT_PRICE_PER_M


class EconomicNetBenefitModel:
    """
    Computes true economic return-on-investment (ROI) of prompt compression.
    Accounts for:
    - Token savings on standard vs cached prompt tiers.
    - Compression execution CPU overhead.
    - Downstream retry costs caused by failed task accuracy or syntax corruption.
    """

    @classmethod
    def evaluate_net_cost(
        cls,
        model_name: str,
        original_tokens: int,
        compressed_tokens: int,
        compression_latency_ms: float,
        task_success_rate: float,
        is_cache_hit: bool = False,
        cpu_cost_per_ms: float = 0.00000002,  # Typical serverless CPU cost per ms
    ) -> Dict[str, Any]:
        input_price_per_m = MODEL_INPUT_PRICE_PER_M.get(model_name, 0.15)
        # OpenAI Prompt Caching offers 50% discount on cache hit
        effective_price_per_m = input_price_per_m * (0.5 if is_cache_hit else 1.0)

        # Baseline cost (raw prompt sent uncompressed)
        baseline_cost_usd = (original_tokens / 1_000_000.0) * effective_price_per_m

        # Compression direct API cost
        direct_comp_cost_usd = (compressed_tokens / 1_000_000.0) * effective_price_per_m

        # Compute cost for running LinUCB on CPU
        compute_cost_usd = compression_latency_ms * cpu_cost_per_ms

        # Expected retry cost if task fails and must be re-run
        failure_rate = max(0.0, 1.0 - task_success_rate)
        expected_retry_cost_usd = failure_rate * baseline_cost_usd

        # Total composite cost
        total_effective_cost_usd = direct_comp_cost_usd + compute_cost_usd + expected_retry_cost_usd

        net_savings_usd = baseline_cost_usd - total_effective_cost_usd
        net_roi_pct = (net_savings_usd / max(1e-9, baseline_cost_usd)) * 100.0

        return {
            "model": model_name,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_reduction_pct": round((1.0 - compressed_tokens / max(1, original_tokens)) * 100.0, 2),
            "baseline_cost_usd": round(baseline_cost_usd, 7),
            "total_effective_cost_usd": round(total_effective_cost_usd, 7),
            "net_savings_usd": round(net_savings_usd, 7),
            "net_roi_pct": round(net_roi_pct, 2),
            "is_net_profitable": net_savings_usd > 0,
        }

    @classmethod
    def compute_latency_percentiles(cls, latencies_us: List[float]) -> Dict[str, float]:
        """Calculates p50, p90, p95, p99 routing latency distributions."""
        if not latencies_us:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(latencies_us)
        return {
            "p50_us": round(float(np.percentile(arr, 50)), 2),
            "p90_us": round(float(np.percentile(arr, 90)), 2),
            "p95_us": round(float(np.percentile(arr, 95)), 2),
            "p99_us": round(float(np.percentile(arr, 99)), 2),
        }
