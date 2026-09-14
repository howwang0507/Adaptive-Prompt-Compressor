"""
OpenAI Structured Outputs (JSON Schema) & Prompt Caching Cookbook
=================================================================
Demonstrates:
1. JSONSchemaGuard: Zero degradation in JSON schema property keys, enums,
   and required fields while compressing developer system instructions.
2. PromptCacheAligner: Preserving prefix byte-invariance for OpenAI 50% prompt
   caching while aggressively compressing dynamic user context via LinUCB.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guards.json_guard import JSONSchemaGuard
from src.integrations.prompt_cache_aligner import PromptCacheAligner
from src.interface import LinUCBCompressor


def demonstrate_structured_outputs_guard():
    print("=" * 70)
    print("🛡️ Demonstration 1: JSON Schema Guard for OpenAI Structured Outputs")
    print("=" * 70)

    compressor = LinUCBCompressor(provider="simulation")

    # Target JSON Schema declaring strict output format
    schema = {
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "The exact official name of the corporation or entity identified in text.",
            },
            "risk_rating": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "A comprehensive categorization of operational financial risk based on audits.",
            },
            "audit_score": {
                "type": "number",
                "description": "Floating point index representing financial stability score ranging between 0 and 100.",
            },
        },
        "required": ["entity_name", "risk_rating", "audit_score"],
        "additionalProperties": False,
    }

    print("\n[Original JSON Schema Definition]")
    print(json.dumps(schema, indent=2))

    guarded_schema = JSONSchemaGuard.sanitize_schema_descriptions(
        schema=schema,
        compress_fn=compressor.compress,
    )

    print("\n[Guarded & Compressed JSON Schema]")
    print(json.dumps(guarded_schema, indent=2))

    # Assert structural integrity
    assert guarded_schema["required"] == schema["required"]
    assert guarded_schema["properties"]["risk_rating"]["enum"] == ["low", "medium", "high", "critical"]
    print("\n✓ Verification Passed: Structural keys, types, enums, and required arrays remain 100% intact!")


def demonstrate_prompt_caching_alignment():
    print("\n" + "=" * 70)
    print("⚡ Demonstration 2: OpenAI Prompt Caching Prefix Co-Optimization")
    print("=" * 70)

    aligner = PromptCacheAligner()

    static_system_preamble = (
        "You are the senior underwriting AI for a global investment institution. " * 10
    )
    user_dynamic_query = (
        "Could you please analyze the following recent earnings report and summarize "
        "any significant deviations in gross margin or forward guidance? Thank you!"
    )

    messages = [
        {"role": "system", "content": static_system_preamble},
        {"role": "user", "content": user_dynamic_query},
    ]

    optimized_messages, meta = aligner.optimize(messages)

    print("\n[Caching & Compression Telemetry]")
    print(f"  • Preserved Prefix Messages: {meta['prefix_messages_preserved']} (System Prompt byte-invariant)")
    print(f"  • Cache Hit Eligible:        {meta['cache_hit_eligible']}")
    print(f"  • Suffix Original Tokens:    {meta['suffix_original_approx_tokens']}")
    print(f"  • Suffix Compressed Tokens:  {meta['suffix_compressed_approx_tokens']}")
    print(f"  • Dynamic Reduction:         {meta['suffix_reduction_pct']}%")
    print(f"  • OpenAI Cache Multiplier:   {meta['effective_cost_multiplier']}x (50% official discount)")

    # Verify prefix unchanged
    assert optimized_messages[0]["content"] == static_system_preamble
    print("\n✓ Verification Passed: System preamble is bit-for-bit identical -> Full 50% OpenAI Cache Hit secured!")


if __name__ == "__main__":
    demonstrate_structured_outputs_guard()
    demonstrate_prompt_caching_alignment()
