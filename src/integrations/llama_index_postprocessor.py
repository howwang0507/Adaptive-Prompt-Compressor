"""
LlamaIndex Integration: Adaptive Contextual NodePostprocessor.
Allows developers to plug Adaptive-Prompt-Compressor directly into LlamaIndex
retrieval pipelines to dynamically compress retrieved Nodes prior to response synthesis.
"""

from typing import Any, List, Optional
from src.interface import LinUCBCompressor


class AdaptiveNodePostprocessor:
    """
    LlamaIndex-compatible NodePostprocessor.
    Takes a list of retrieved NodeWithScore objects or raw text snippets,
    evaluates their context complexity via LinUCB Bandits, prunes non-essential
    filler tokens, and outputs compressed Nodes preserving semantic retrieval fidelity.
    """

    def __init__(
        self,
        compressor: Optional[LinUCBCompressor] = None,
        target_token_reduction: float = 0.35,
    ):
        self.compressor = compressor or LinUCBCompressor(provider="simulation")
        self.target_token_reduction = target_token_reduction

    def postprocess_nodes(
        self,
        nodes: List[Any],
        query_str: Optional[str] = None,
    ) -> List[Any]:
        """
        Compresses text inside each retrieved node.
        Works seamlessly whether nodes are LlamaIndex NodeWithScore objects or dicts/strings.
        """
        processed_nodes = []
        for node in nodes:
            # Handle standard LlamaIndex NodeWithScore or BaseNode
            if hasattr(node, "node") and hasattr(node.node, "get_content"):
                raw_text = node.node.get_content()
                compressed_text, strategy, meta = self.compressor.compress(raw_text)
                node.node.set_content(compressed_text)
                if not hasattr(node.node, "metadata"):
                    node.node.metadata = {}
                node.node.metadata["compression_strategy"] = strategy
                processed_nodes.append(node)
            elif isinstance(node, dict) and "text" in node:
                compressed_text, strategy, _ = self.compressor.compress(node["text"])
                processed_nodes.append({**node, "text": compressed_text, "strategy": strategy})
            elif isinstance(node, str):
                compressed_text, _, _ = self.compressor.compress(node)
                processed_nodes.append(compressed_text)
            else:
                processed_nodes.append(node)

        return processed_nodes
