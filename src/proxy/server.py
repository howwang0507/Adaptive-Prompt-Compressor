"""
OpenAI-Compatible Reverse Proxy Gateway.
Allows ANY programming language (cURL, Python, TypeScript, Go, Java, Rust)
to route OpenAI requests through Adaptive-Prompt-Compressor simply by setting:
    OPENAI_BASE_URL="http://localhost:8000/v1"
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from src.interface import LinUCBCompressor
from src.guards.json_guard import JSONSchemaGuard
from src.integrations.prompt_cache_aligner import PromptCacheAligner


class OpenAIProxyGateway:
    """
    Core engine for the OpenAI-compatible proxy gateway.
    Handles transparent intercept, cache alignment, JSON guard, and Bandit compression.
    """

    def __init__(
        self,
        compressor: Optional[LinUCBCompressor] = None,
        align_caching: bool = True,
        enable_json_guard: bool = True,
    ):
        self.compressor = compressor or LinUCBCompressor(provider="simulation")
        self.cache_aligner = PromptCacheAligner(compressor=self.compressor) if align_caching else None
        self.enable_json_guard = enable_json_guard

    def process_chat_completion_request(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Intercepts chat completion payload, optimizes message prompts,
        and attaches compression telemetry.
        """
        messages: List[Dict[str, str]] = payload.get("messages", [])
        model = payload.get("model", "gpt-4o-mini")
        t0 = time.perf_counter()

        orig_token_estimate = sum(len(m.get("content", "").split()) for m in messages if isinstance(m.get("content"), str))

        # 1. Apply Prompt Cache Alignment if enabled
        if self.cache_aligner:
            optimized_messages, cache_meta = self.cache_aligner.optimize(messages, model=model)
        else:
            optimized_messages = []
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    comp, _, _ = self.compressor.compress(content)
                    optimized_messages.append({**m, "content": comp})
                else:
                    optimized_messages.append(m)
            cache_meta = {}

        # 2. Apply JSON Schema Guard if response_format declared
        response_format = payload.get("response_format")
        if self.enable_json_guard and response_format and response_format.get("type") == "json_schema":
            schema_data = response_format.get("json_schema", {}).get("schema", {})
            if schema_data:
                guarded_schema = JSONSchemaGuard.sanitize_schema_descriptions(schema_data, self.compressor.compress)
                payload["response_format"]["json_schema"]["schema"] = guarded_schema

        comp_token_estimate = sum(len(m.get("content", "").split()) for m in optimized_messages if isinstance(m.get("content"), str))
        latency_us = (time.perf_counter() - t0) * 1_000_000

        telemetry = {
            "model": model,
            "original_tokens_approx": orig_token_estimate,
            "compressed_tokens_approx": comp_token_estimate,
            "tokens_saved": max(0, orig_token_estimate - comp_token_estimate),
            "savings_ratio_pct": (
                round((1.0 - comp_token_estimate / max(1, orig_token_estimate)) * 100.0, 2)
                if orig_token_estimate > 0
                else 0.0
            ),
            "proxy_overhead_us": round(latency_us, 1),
            "cache_alignment": cache_meta,
        }

        updated_payload = {**payload, "messages": optimized_messages}
        return updated_payload, telemetry

    def simulate_upstream_response(self, processed_payload: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates standard OpenAI /v1/chat/completions response object with telemetry headers.
        """
        return {
            "id": f"chatcmpl-proxy-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": processed_payload.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This response was routed through the Adaptive-Prompt-Compressor reverse proxy gateway.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": telemetry["compressed_tokens_approx"],
                "completion_tokens": 18,
                "total_tokens": telemetry["compressed_tokens_approx"] + 18,
            },
            "compression_meta": telemetry,
        }
