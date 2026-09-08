from src.interface import LinUCBCompressor


def test_openai_compressor_init():
    compressor = LinUCBCompressor(provider="openai", model_name="gpt-4o-mini")
    assert compressor.provider in ["openai", "openai (offline simulation)"]
    assert len(compressor.strategies) == 3


def test_openai_compressor_code_routing():
    compressor = LinUCBCompressor(provider="openai", model_name="gpt-4o-mini")
    code_prompt = "def hello_world():\n    return 'hello world'\n"
    compressed, strategy, meta = compressor.compress(code_prompt)
    assert isinstance(compressed, str)
    assert strategy in compressor.strategies
    assert meta["arm"] in [0, 1, 2]
    # Code syntax should be preserved
    assert "def hello_world" in compressed


def test_openai_compressor_chat_routing():
    compressor = LinUCBCompressor(provider="openai", model_name="gpt-4o-mini")
    chat_prompt = "Could you please summarize this long document for me?"
    compressed, strategy, meta = compressor.compress(chat_prompt)
    assert isinstance(compressed, str)
    assert len(compressed) > 0
