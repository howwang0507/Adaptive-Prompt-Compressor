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
        original_input_tokens: int,
        compressed_input_tokens: int,
        output_tokens: int = 150,
        compression_latency_ms: float = 0.08,
        verification_latency_ms: float = 0.02,
        task_success_rate: float = 1.0,
        cache_hit_ratio: float = 0.0,
        cpu_cost_per_ms: float = 0.00000002,  # Standard AWS Lambda / Cloud Run rate
    ) -> Dict[str, Any]:
        """
        Calculates realistic end-to-end economic net benefit:
        - Input token rate factoring prompt cache discount (50%).
        - Output generation token cost.
        - LinUCB decision + structural verification compute overhead.
        - Downstream retry costs on task accuracy failure.
        """
        input_price_per_m = MODEL_INPUT_PRICE_PER_M.get(model_name, 0.15)
        output_price_per_m = input_price_per_m * 4.0  # Typically 3x-4x input price

        # Effective input price with cache discount
        effective_input_price_per_m = input_price_per_m * (1.0 - 0.5 * cache_hit_ratio)

        # Baseline cost (raw prompt without compression)
        baseline_input_cost = (original_input_tokens / 1_000_000.0) * effective_input_price_per_m
        output_cost = (output_tokens / 1_000_000.0) * output_price_per_m
        baseline_total_cost_usd = baseline_input_cost + output_cost

        # Compressed input cost
        compressed_input_cost = (compressed_input_tokens / 1_000_000.0) * effective_input_price_per_m

        # Total compute overhead (Bandit routing + Guard verification)
        total_compute_latency_ms = compression_latency_ms + verification_latency_ms
        compute_cost_usd = total_compute_latency_ms * cpu_cost_per_ms

        # Expected retry cost if downstream task or AST fails
        failure_rate = max(0.0, 1.0 - task_success_rate)
        expected_retry_cost_usd = failure_rate * baseline_total_cost_usd

        # Total composite cost
        total_effective_cost_usd = compressed_input_cost + output_cost + compute_cost_usd + expected_retry_cost_usd

        net_savings_usd = baseline_total_cost_usd - total_effective_cost_usd
        net_roi_pct = (net_savings_usd / max(1e-9, baseline_total_cost_usd)) * 100.0

        return {
            "model": model_name,
            "original_input_tokens": original_input_tokens,
            "compressed_input_tokens": compressed_input_tokens,
            "output_tokens": output_tokens,
            "token_reduction_pct": round((1.0 - compressed_input_tokens / max(1, original_input_tokens)) * 100.0, 2),
            "baseline_total_cost_usd": round(baseline_total_cost_usd, 7),
            "total_effective_cost_usd": round(total_effective_cost_usd, 7),
            "compute_cost_usd": round(compute_cost_usd, 7),
            "expected_retry_cost_usd": round(expected_retry_cost_usd, 7),
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
