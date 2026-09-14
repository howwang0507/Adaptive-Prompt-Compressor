"""
Unit tests for Proxy Gateway, LlamaIndex NodePostprocessor, and Whitepaper features.
"""

from src.proxy.server import OpenAIProxyGateway
from src.integrations.llama_index_postprocessor import AdaptiveNodePostprocessor


def test_proxy_gateway_processing():
    gateway = OpenAIProxyGateway()
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a quick sorting function in Python."},
        ],
    }
    processed_payload, telemetry = gateway.process_chat_completion_request(payload)

    assert "messages" in processed_payload
    assert len(processed_payload["messages"]) == 2
    assert telemetry["model"] == "gpt-4o-mini"
    assert telemetry["proxy_overhead_us"] > 0


def test_llama_index_node_postprocessor():
    postprocessor = AdaptiveNodePostprocessor()
    nodes = [
        {"text": "This is a retrieved passage describing corporate financial performance in 2025."},
        "A raw string retrieved passage from vector index.",
    ]

    processed = postprocessor.postprocess_nodes(nodes)
    assert len(processed) == 2
    assert isinstance(processed[0], dict)
    assert "text" in processed[0]
