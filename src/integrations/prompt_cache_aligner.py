"""
OpenAI Prompt Caching Prefix Alignment Optimizer.
OpenAI caches prompts with prefixes >= 1,024 tokens, offering a 50% discount on cache hits.
This module aligns prompt boundaries so that static prefixes (system instructions, common few-shots)
remain byte-for-byte identical, stacking the 50% OpenAI Prompt Caching discount with
LinUCB contextual compression on the dynamic suffix.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.interface import LinUCBCompressor


class PromptCacheAligner:
    """
    Splits conversational message histories or documents into:
      1. Static Cached Prefix (immutable, eligible for OpenAI 50% Prompt Caching)
      2. Dynamic User/RAG Suffix (compressed via LinUCB Bandits)
    """

    def __init__(
        self,
        compressor: Optional[LinUCBCompressor] = None,
        min_prefix_tokens: int = 1024,
    ):
        self.compressor = compressor or LinUCBCompressor(provider="simulation")
        self.min_prefix_tokens = min_prefix_tokens

    def partition_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Partitions message sequence into:
          - cached_prefix: Initial system messages and static context (not mutated)
          - dynamic_suffix: User queries, dynamic tool calls, and volatile contexts
        """
        if not messages:
            return [], []

        prefix_msgs: List[Dict[str, str]] = []
        suffix_msgs: List[Dict[str, str]] = []

        # System messages and invariant preambles form the cached prefix
        in_prefix = True
        for msg in messages:
            role = msg.get("role", "")
            if in_prefix and role in ("system", "developer"):
                prefix_msgs.append(msg)
            else:
                in_prefix = False
                suffix_msgs.append(msg)

        # If there are no system messages, the first non-system message stays if requested
        if not prefix_msgs and len(suffix_msgs) > 1:
            prefix_msgs.append(suffix_msgs[0])
            suffix_msgs = suffix_msgs[1:]

        return prefix_msgs, suffix_msgs

    def optimize(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Applies Prefix-Invariant Caching alignment + LinUCB compression.
        Returns:
            (optimized_messages, telemetry_metadata)
        """
        prefix_msgs, suffix_msgs = self.partition_messages(messages)

        compressed_suffix = []
        suffix_orig_tokens = 0
        suffix_comp_tokens = 0
        strategies_used = []

        for msg in suffix_msgs:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                orig_len = len(content.split())
                comp_text, strategy, _ = self.compressor.compress(content)
                comp_len = len(comp_text.split())

                suffix_orig_tokens += orig_len
                suffix_comp_tokens += comp_len
                strategies_used.append(strategy)
                compressed_suffix.append({**msg, "content": comp_text})
            else:
                compressed_suffix.append(msg)

        prefix_tokens = sum(len(m.get("content", "").split()) for m in prefix_msgs)
        cache_eligible = prefix_tokens >= (self.min_prefix_tokens // 4)

        metadata = {
            "prefix_messages_preserved": len(prefix_msgs),
            "prefix_approx_tokens": prefix_tokens,
            "cache_hit_eligible": cache_eligible,
            "suffix_original_approx_tokens": suffix_orig_tokens,
            "suffix_compressed_approx_tokens": suffix_comp_tokens,
            "suffix_reduction_pct": (
                round((1.0 - suffix_comp_tokens / max(1, suffix_orig_tokens)) * 100.0, 2)
                if suffix_orig_tokens > 0
                else 0.0
            ),
            "strategies_applied": strategies_used,
            "effective_cost_multiplier": 0.5 if cache_eligible else 1.0,
        }

        return prefix_msgs + compressed_suffix, metadata
