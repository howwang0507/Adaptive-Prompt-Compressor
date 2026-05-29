from typing import Dict, Optional
from langchain_core.runnables import RunnableSerializable
from src.interface import LinUCBCompressor


class AdaptiveCompressionWrapper(RunnableSerializable):
    """
    LangChain Integration for the Adaptive Prompt Compressor.
    Allows seamless inclusion of dynamic compression in any LCEL pipeline.
    """

    compressor: LinUCBCompressor

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.compressor = LinUCBCompressor(api_key=api_key)

    def invoke(self, input: str, config: Optional[Dict] = None) -> str:
        # Automatically compress the input prompt before passing it to the LLM
        compressed_text, strategy, _ = self.compressor.compress(input)
        print(f"--- Adaptive Routing: Selected {strategy} ---")
        return compressed_text


# Example Usage:
# chain = AdaptiveCompressionWrapper() | ChatOpenAI()
# chain.invoke("Your long prompt here")
