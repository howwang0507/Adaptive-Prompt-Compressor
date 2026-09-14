# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- **Reasoning Model Support**: Optimized compression routines for OpenAI `o1`, `o1-mini`, and `o3-mini`.
- **Streaming Compression Middleware**: Chunked prompt preprocessing with zero impact on Time-To-First-Token (TTFT).

## [1.4.2] - 2026-09-14

### Fixed & Hardened
- **Structural Guard Counter-Example Elimination (`src/guards/structural_guard.py`)**:
  - **CJK-Safe Digit Boundary**: Replaced ASCII `\b` with lookarounds `(?<!\d)` and `(?!\d)`, ensuring numbers surrounded by continuous Chinese characters (e.g. `保留30天` vs `保留300天`) are accurately extracted and defended.
  - **Currency Symbol Verification**: Added strict currency symbol binding and equality verification (`Pay $100` vs `Pay €100`), catching cross-currency corruptions.
  - **Negative Sign & Operator Matching**: Added signed number support `[-+]?` and single equality operator `=`, preventing sign stripping (`Keep x = -30` vs `Keep x = 30`).
  - **Negation Scope & Target Binding**: Implemented clause-level negation target extraction to prevent negation transfer and polarity inversion (`Do not delete A. Delete B.` vs `Delete A. Do not delete B.`).
- **Robust RAG Sentence Segmentation (`src/rag/evidence_compressor.py`)**:
  - **Trailing Sentence Retention**: Ensured unpunctuated final sentences (`Final evidence without punctuation`) are strictly preserved and never dropped before relevance ranking.
  - **Decimal & Version Protection**: Masked floating point numbers (`3.14`, `0.0025`) and version identifiers (`v1.4.2`) to prevent premature mid-number sentence splitting.
  - **URL Protection**: Preserved full URLs with trailing punctuation handling (`https://example.com/report`).

## [1.4.1] - 2026-09-14

### Fixed & Hardened
- **Structural Guard Counter-Example Elimination (`src/guards/structural_guard.py`)**:
  - Implemented exact word-boundary matching (`\bnot\b`) preventing false positives from words like `nothing`.
  - Added character-level Chinese substring matching for negations (`不得`, `禁止`, `切勿`) eliminating CJK tokenization misses.
  - Bound comparison operators, numbers, and units into atomic tuples (`<= 30 kg` vs `> 30 g`, `30 days` vs `300 days`).
  - Implemented real markdown code block placeholder isolation, preserving python code indentations byte-for-byte.
- **Strict Barrier Constrained Reward Formulation (`src/utils.py`)**:
  - Enforced that any relative quality degradation violating tolerance receives a strictly negative reward (`-barrier_penalty - lambda * violation`), eliminating positive-reward loopholes for low-quality outputs.
- **RAG Mandatory Citation Retention (`src/rag/evidence_compressor.py`)**:
  - Made citations (`[Doc 1]`, `[Doc 2]`) a mandatory retention set, and bound trailing citation tags to preceding sentences with post-verification fallback.
- **Economic Net Benefit Realistic Modeling (`src/economics/economic_model.py`)**:
  - Factored in output tokens, verification CPU latency, and retry costs into true composite cost calculations.
- **Academic Benchmark Scope Alignment**:
  - Updated README and scripts to accurately label HumanEval as Code Context AST Preservation and proxy comparisons as Static Token-Level Pruning Proxy.

## [1.4.0] - 2026-09-14

### Added
- **Block-Level Structural & Constraint Guard (`src/guards/structural_guard.py`)**:
  - Python indentation and syntax isolation.
  - Mandatory preservation of critical negation keywords (`not`, `never`, `不得`, `禁止`) and numerical constraint tokens.
  - Automatic safe fallback to original prompt upon syntax or constraint degradation.
- **Constrained Bandit Objective & Closed-Loop Feedback**:
  - Lagrangian-penalized reward function `calculate_constrained_reward` in `src/utils.py`.
  - Downstream accuracy reporting method `response.report_feedback()` in `src/integrations/openai_client.py`.
- **Evidence-Preserving RAG Compressor (`src/rag/evidence_compressor.py`)**:
  - Sentence-level query-relevance ranking and citation anchor preservation `[Doc 1]`.
  - Character-safe multilingual sentence segmentation (Traditional & Simplified Chinese support).
- **Comprehensive Economic Net Benefit Model (`src/economics/economic_model.py`)**:
  - Multi-factor ROI evaluation accounting for token discounts, serverless CPU compute overhead, retry penalties, and latency percentiles (p50/p95).
- **Academic Benchmark Rigor & Ground Truth Alignment**:
  - Clarified benchmark scopes to accurately reflect code context AST preservation and proxy token pruning.

## [1.3.0] - 2026-09-14

### Added
- **OpenAI-Compatible Reverse Proxy Gateway (`src/proxy/server.py`)**:
  - Transparent interceptor allowing any framework or language (cURL, Go, TypeScript) to use compression by setting `OPENAI_BASE_URL`.
- **LlamaIndex Adaptive NodePostprocessor (`src/integrations/llama_index_postprocessor.py`)**:
  - Pluggable postprocessor dynamically compressing retrieved text nodes in RAG synthesis.
- **GSM8K Reasoning & Mathematical Entity Preservation Benchmark (`benchmarks/eval_gsm8k.py`)**:
  - Validated 100% entity fidelity and numerical consistency under LinUCB contextual compression.
- **Technical Whitepaper & Formal Architecture Specification (`docs/WHITE_PAPER.md`)**:
  - Full formulation covering Sherman-Morrison rank-1 updates, regret bounds, and double-cost-reduction analysis.
- **GitHub Profile Showcase Hub (`profile/README.md`)**:
  - Personalized profile documentation highlighting architecture achievements.
- **Automated Ecosystem Tests (`tests/test_v13_ecosystem.py`)**:
  - Full coverage for proxy gateway and LlamaIndex adapters.

## [1.2.0] - 2026-09-14

### Added
- **JSON Schema Guard for OpenAI Structured Outputs (`src/guards/json_guard.py`)**:
  - Automatically isolates and protects `properties`, `type`, `required`, and `enum` fields.
  - Recursively compresses verbose `description` strings without compromising JSON parsing validity.
- **OpenAI Prompt Caching Prefix Co-Optimization (`src/integrations/prompt_cache_aligner.py`)**:
  - Implemented boundary partitioner ensuring static system instructions remain 100% byte-for-byte invariant to maximize OpenAI's 50% Prompt Caching discount.
  - Dynamically routes LinUCB bandit compression across user and retrieved context suffixes.
- **Interactive Structured Outputs & Prompt Caching Cookbook**:
  - Added `examples/openai_structured_outputs_and_caching_demo.py` showcasing end-to-end telemetry and verification.
- **Comprehensive Unit Testing Suite**:
  - Added `tests/test_v12_features.py` covering schema integrity and prefix-invariance.
- **High-Throughput Async Batch Processing**:
  - Added `AsyncLinUCBCompressor` batch processing support and cookbook demo in `examples/async_batch_compression_demo.py`.

## [1.1.2] - 2026-09-14

### Added
- **Tool / Function Calling Compression Cookbook**:
  - Added `examples/openai_tool_calling_compression.py` demonstrating zero-degradation system prompt compression with schema verification during tool usage.
- **Enhanced Open-Source Contributing Guidelines**:
  - Rewrote and enriched `CONTRIBUTING.md` with Conventional Commits, branch workflows, and developer quality checks.
- **Benchmark Artifacts Management**:
  - Structured empirical benchmark outputs into `benchmarks/results/`.

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
