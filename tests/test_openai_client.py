from unittest.mock import MagicMock
from src.integrations.openai_client import OpenAICompressorClient, wrap_openai_client
from src.interface import LinUCBCompressor


def test_openai_compressor_client_mock():
    # Setup mock OpenAI client
    mock_raw_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock answer"
    mock_raw_client.chat.completions.create.return_value = mock_response

    compressor = LinUCBCompressor(provider="simulation")
    client = OpenAICompressorClient(client=mock_raw_client, compressor=compressor)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please write a function to calculate factorial."},
    ]

    res = client.chat.completions.create(model="gpt-4o-mini", messages=messages)

    assert res.choices[0].message.content == "Mock answer"
    assert hasattr(res, "compression_meta")
    assert "char_savings_pct" in res.compression_meta
    assert "strategies" in res.compression_meta


def test_wrap_openai_client_helper():
    mock_raw_client = MagicMock()
    client = wrap_openai_client(mock_raw_client)
    assert isinstance(client, OpenAICompressorClient)
