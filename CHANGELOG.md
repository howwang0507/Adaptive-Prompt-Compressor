# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- **OpenAI Structured Outputs (JSON Schema)**: Guardrails to compress user payloads without breaking JSON schema structural constraints.
- **OpenAI Prompt Cache Prefix Alignment**: Preserve exact prompt prefix boundaries to stack LinUCB compression with OpenAI's 50% prompt caching discount.
- **Reasoning Model Support**: Optimized compression routines for OpenAI `o1`, `o1-mini`, and `o3-mini`.

## [1.1.0] - 2026-09-12

### Added
- **OpenAI Python SDK 1-Line Drop-in Wrapper (`wrap_openai_client`)**:
  - Automatically hooks into `chat.completions.create`.
  - Transparent prompt token compression prior to network dispatch.
- **Accurate Token Counting & Cost Telemetry**:
  - Integrated `tiktoken` with model-aware tokenizers (`o200k_base`, `cl100k_base`).
  - Added live `compression_meta` to API responses (`token_savings_pct`, `char_savings_pct`, `tokens_saved`, `est_cost_savings_usd`).
  - Built-in pricing table for `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, and `o3-mini`.
- **Multi-Version CI/CD Matrix**:
  - GitHub Actions testing against Python 3.10, 3.11, and 3.12 with clean 100% passing status.
- **Pytest Isolation**:
  - Configured `pyproject.toml` with explicit `testpaths = ["tests"]` to isolate automated CI runs from ad-hoc manual scripts.

### Changed
- Refined `tests/test_openai_client.py` with end-to-end token reduction and USD cost assertions.
- Cleaned up root directory temporary artifacts (`test_fastmcp*.py`, `fix_agent.py`).

### Fixed
- Resolved Pytest fixture errors on standalone script execution.
- Resolved type consistency in `LinUCB` arm selection for NumPy array inputs.

## [1.0.0] - 2026-08-15

### Added
- Initial release of Adaptive Prompt Compressor.
- Contextual Multi-Armed Bandit engine using LinUCB with Sherman-Morrison $O(d^2)$ rank-1 updates.
- Abstract Syntax Tree (AST) guard ensuring 100% Python syntax preservation.
- Model Context Protocol (MCP) server support.
- Streamlit interactive tuning and fleet telemetry visualization.
