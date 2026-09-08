"""
OpenAI SDK Drop-in Wrapper with Adaptive Prompt Compression.
Allows developers to wrap their standard `openai.OpenAI` client in one line:

    from openai import OpenAI
    from src.integrations.openai_client import wrap_openai_client

    client = wrap_openai_client(OpenAI())
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "..."}]
    )
    print(response.compression_meta)
"""

from typing import Any, List, Optional
import openai
from src.interface import LinUCBCompressor


class OpenAICompressorClient:
    """
    Drop-in wrapper for `openai.OpenAI` that applies task-aware LinUCB compression
    to incoming prompts before transmitting them to OpenAI's API.
    """

    def __init__(
        self,
        client: Optional[openai.OpenAI] = None,
        api_key: Optional[str] = None,
        compressor: Optional[LinUCBCompressor] = None,
        compress_system_prompts: bool = False,
    ):
        self._raw_client = client or openai.OpenAI(api_key=api_key)
        self.compressor = compressor or LinUCBCompressor(provider="openai")
        self.compress_system_prompts = compress_system_prompts
        self.chat = _ChatWrapper(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw_client, name)


class _ChatWrapper:
    def __init__(self, parent: OpenAICompressorClient):
        self._parent = parent
        self.completions = _CompletionsWrapper(parent)


class _CompletionsWrapper:
    def __init__(self, parent: OpenAICompressorClient):
        self._parent = parent

    def create(self, *args, **kwargs):
        messages: List[dict] = kwargs.get("messages", [])
        compressed_messages = []
        total_original_chars = 0
        total_compressed_chars = 0
        strategies_used = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Compress string content for user role (and system if enabled)
            if isinstance(content, str) and (
                role == "user" or (role == "system" and self._parent.compress_system_prompts)
            ):
                orig_len = len(content)
                comp_text, strategy, _ = self._parent.compressor.compress(content)
                comp_len = len(comp_text)

                total_original_chars += orig_len
                total_compressed_chars += comp_len
                strategies_used.append(strategy)

                compressed_messages.append({**msg, "content": comp_text})
            else:
                compressed_messages.append(msg)

        kwargs["messages"] = compressed_messages
        response = self._parent._raw_client.chat.completions.create(*args, **kwargs)

        # Attach compression analytics
        savings_pct = (
            max(0.0, (1.0 - total_compressed_chars / max(1, total_original_chars)) * 100.0)
            if total_original_chars > 0
            else 0.0
        )
        setattr(
            response,
            "compression_meta",
            {
                "strategies": strategies_used,
                "char_savings_pct": round(savings_pct, 2),
                "original_chars": total_original_chars,
                "compressed_chars": total_compressed_chars,
            },
        )
        return response


def wrap_openai_client(
    client: Optional[openai.OpenAI] = None,
    compressor: Optional[LinUCBCompressor] = None,
    compress_system_prompts: bool = False,
) -> OpenAICompressorClient:
    """Helper function to wrap an existing OpenAI client with 1 line of code."""
    return OpenAICompressorClient(
        client=client,
        compressor=compressor,
        compress_system_prompts=compress_system_prompts,
    )
