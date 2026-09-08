import time
import asyncio
import logging
from functools import wraps

# Setup a dedicated telemetry logger
logger = logging.getLogger("TelemetryDB")
logger.setLevel(logging.INFO)
# In production, this would log to a time-series DB like Prometheus or InfluxDB
# For now, we use a structured log file
fh = logging.FileHandler("results/system_telemetry.log")
fh.setFormatter(logging.Formatter('{"timestamp": "%(asctime)s", "metric": "%(message)s"}'))
logger.addHandler(fh)

# --- Metric Stores ---
class MetricsStore:
    redis_sync_latencies = []
    guardrail_triggers = 0
    ast_failures = 0
    ttft_improvements = []

def track_redis_latency(func):
    """Decorator to measure and log Redis O(d^2) parameter sync latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        result = func(*args, **kwargs)
        latency_us = (time.perf_counter() - start_t) * 1_000_000
        MetricsStore.redis_sync_latencies.append(latency_us)
        logger.info(f"Redis_Sync_Latency_us: {latency_us:.2f}")
        return result
    return wrapper

def track_guardrail(trigger_type="FREEZE_MASK"):
    """Logs security guardrail activations to prove system stability."""
    MetricsStore.guardrail_triggers += 1
    logger.info(f"Guardrail_Trigger: {trigger_type}")

def track_ast_failure():
    """Logs AST syntax failures caught during technical tasks."""
    MetricsStore.ast_failures += 1
    logger.info("AST_Validation_Failure")

def log_ttft_improvement(raw_ttft_ms, compressed_ttft_ms):
    """Logs the TTFT improvement ratio after prompt compression."""
    improvement = ((raw_ttft_ms - compressed_ttft_ms) / raw_ttft_ms) * 100 if raw_ttft_ms > 0 else 0
    MetricsStore.ttft_improvements.append(improvement)
    logger.info(f"TTFT_Improvement_pct: {improvement:.2f}")

