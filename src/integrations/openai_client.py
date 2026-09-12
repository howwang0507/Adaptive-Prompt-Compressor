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

from typing import Any, Dict, List, Optional
import openai
from src.interface import LinUCBCompressor

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Pricing per million tokens (input tokens) as of 2026/latest OpenAI rates
MODEL_INPUT_PRICE_PER_M: Dict[str, float] = {
    "gpt-4o": 2.50,
    "gpt-4o-2024-08-06": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4o-mini-2024-07-18": 0.15,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
    "o1": 15.00,
    "o1-mini": 3.00,
    "o3-mini": 1.10,
}


def count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    """Exact token counting with tiktoken, falling back to character approximation."""
    if not text:
        return 0
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            try:
                enc = tiktoken.get_encoding("o200k_base")
                return len(enc.encode(text))
            except Exception:
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                    return len(enc.encode(text))
                except Exception:
                    pass
    return max(1, len(text) // 4)


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
        model_name = kwargs.get("model", "gpt-4o-mini")
        compressed_messages = []
        total_original_chars = 0
        total_compressed_chars = 0
        total_original_tokens = 0
        total_compressed_tokens = 0
        strategies_used = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Compress string content for user role (and system if enabled)
            if isinstance(content, str) and (
                role == "user" or (role == "system" and self._parent.compress_system_prompts)
            ):
                orig_len = len(content)
                orig_tokens = count_tokens_tiktoken(content, model=model_name)
                comp_text, strategy, _ = self._parent.compressor.compress(content)
                comp_len = len(comp_text)
                comp_tokens = count_tokens_tiktoken(comp_text, model=model_name)

                total_original_chars += orig_len
                total_compressed_chars += comp_len
                total_original_tokens += orig_tokens
                total_compressed_tokens += comp_tokens
                strategies_used.append(strategy)

                compressed_messages.append({**msg, "content": comp_text})
            else:
                if isinstance(content, str):
                    c_tokens = count_tokens_tiktoken(content, model=model_name)
                    total_original_tokens += c_tokens
                    total_compressed_tokens += c_tokens
                    total_original_chars += len(content)
                    total_compressed_chars += len(content)
                compressed_messages.append(msg)

        kwargs["messages"] = compressed_messages
        response = self._parent._raw_client.chat.completions.create(*args, **kwargs)

        # Attach compression analytics
        char_savings_pct = (
            max(0.0, (1.0 - total_compressed_chars / max(1, total_original_chars)) * 100.0)
            if total_original_chars > 0
            else 0.0
        )
        token_savings_pct = (
            max(0.0, (1.0 - total_compressed_tokens / max(1, total_original_tokens)) * 100.0)
            if total_original_tokens > 0
            else 0.0
        )
        tokens_saved = max(0, total_original_tokens - total_compressed_tokens)

        # Calculate estimated dollar savings
        price_per_m = MODEL_INPUT_PRICE_PER_M.get(model_name, 0.15)
        est_cost_savings_usd = (tokens_saved / 1_000_000.0) * price_per_m

        setattr(
            response,
            "compression_meta",
            {
                "strategies": strategies_used,
                "token_savings_pct": round(token_savings_pct, 2),
                "char_savings_pct": round(char_savings_pct, 2),
                "original_tokens": total_original_tokens,
                "compressed_tokens": total_compressed_tokens,
                "tokens_saved": tokens_saved,
                "est_cost_savings_usd": round(est_cost_savings_usd, 7),
                "model": model_name,
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

