import asyncio
import json
import logging
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

# Import our Hybrid Neural-Bandit Compressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.async_interface import AsyncLinUCBCompressor
from src.telemetry_logger import track_guardrail, log_ttft_improvement

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Server and Compressor
mcp = FastMCP("BuildingMonitorCompressor", host="0.0.0.0", port=9000)

# In production, API key is injected via environment variables
api_key = os.getenv("GEMINI_API_KEY", "")
compressor = AsyncLinUCBCompressor(api_key=api_key, model_name="gemini-1.5-flash")

@mcp.tool("compress_sensor_log", "Compresses voluminous building sensor JSON logs into a dense semantic prompt before LLM analysis.")
async def compress_sensor_log(raw_log_payload: str) -> str:
    """
    Called by the upstream LLM Agent when it receives a massive log dump.
    """
    try:
        # 1. Quick validation (ensure it's actually a log)
        log_data = json.loads(raw_log_payload)
        sensor_id = log_data.get("sensor_id", "UNKNOWN")
        logging.info(f"Received log from sensor: {sensor_id} (Length: {len(raw_log_payload)})")
    except json.JSONDecodeError:
        logging.warning("Payload is not valid JSON. Proceeding with raw text compression.")

    # 2. OOD Guard / Freeze Mask injection for critical alarms
    if "CRITICAL_ALARM" in raw_log_payload or "FIRE" in raw_log_payload:
        raw_log_payload = "FREEZE_MASK: " + raw_log_payload
        track_guardrail("CRITICAL_ALARM_FREEZE")
        logging.info("Critical alarm detected. Applying Guardrail Mask to force Arm 0 (Raw Fidelity).")

    # 3. Execute Async Compression (O(d^2) latency)
    import time
    start_t = time.perf_counter()
    compressed_text, strategy_used, metadata = await compressor.compress(raw_log_payload)
    comp_latency_ms = (time.perf_counter() - start_t) * 1000
    
    # 4. Simulate TTFT Logging
    # Assuming baseline LLM TTFT is linearly proportional to token length (mock for telemetry)
    # E.g. 50ms per 1000 chars
    raw_ttft_mock = len(raw_log_payload) * 0.05
    compressed_ttft_mock = len(compressed_text) * 0.05 + comp_latency_ms
    log_ttft_improvement(raw_ttft_mock, compressed_ttft_mock)
    
    saving_pct = metadata.get("saving_ratio", 0) * 100
    logging.info(f"Compression Complete: Strategy {strategy_used} | Saved: {saving_pct:.1f}%")
    
    # Return the dense context to the LLM
    return compressed_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport instead of stdio")
    args = parser.parse_args()
    
    if args.sse:
        logging.info("Starting MCP Server in SSE mode on port 9000...")
        mcp.run(transport="sse")
    else:
        logging.info("Starting MCP Server in stdio mode...")
        mcp.run()
