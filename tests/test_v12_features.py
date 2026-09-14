"""
Unit tests for JSONSchemaGuard, PromptCacheAligner, and AsyncLinUCBCompressor.
Ensures zero degradation in JSON structure, prefix-invariant prompt caching, and concurrent batching.
"""

import pytest
from src.async_interface import AsyncLinUCBCompressor
from src.guards.json_guard import JSONSchemaGuard
from src.integrations.prompt_cache_aligner import PromptCacheAligner
from src.interface import LinUCBCompressor


def test_json_schema_guard_preserves_structure():
    compressor = LinUCBCompressor(provider="simulation")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The full name of the user or subject."},
            "status": {"type": "string", "enum": ["active", "suspended", "pending"]},
        },
        "required": ["name", "status"],
    }

    guarded = JSONSchemaGuard.sanitize_schema_descriptions(schema, compressor.compress)

    assert guarded["type"] == "object"
    assert guarded["required"] == ["name", "status"]
    assert guarded["properties"]["status"]["enum"] == ["active", "suspended", "pending"]
    assert "description" in guarded["properties"]["name"]


def test_prompt_cache_aligner_preserves_prefix():
    aligner = PromptCacheAligner()
    system_text = "You are an immutable enterprise security compliance assistant."
    user_text = "Please verify whether this password policy fulfills NIST standards."

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    optimized, meta = aligner.optimize(messages)

    assert len(optimized) == 2
    assert optimized[0]["content"] == system_text  # Exactly unchanged for OpenAI prompt caching
    assert optimized[0]["role"] == "system"
    assert meta["prefix_messages_preserved"] == 1


@pytest.mark.anyio
async def test_async_batch_compression():
    async_compressor = AsyncLinUCBCompressor()
    prompts = [
        "First prompt for testing async pipeline.",
        "Second prompt for testing batch latency.",
    ]
    results = await async_compressor.compress_batch(prompts)
    assert len(results) == 2
    for r in results:
        assert "response" in r or "strategy_used" in r
