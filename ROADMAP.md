# Project Roadmap & Technical Milestones 🗺️

This document outlines the strategic engineering roadmap for **Adaptive-Prompt-Compressor**, designed to align with evolving LLM infrastructure, OpenAI models, and enterprise latency budgets.

---

## 📅 Version Milestones

### ✅ v1.0.0 — Contextual Bandit Foundation (Released)
- [x] LinUCB algorithm implementation with Sherman-Morrison $O(d^2)$ incremental inverse updates.
- [x] Multi-provider simulation engine (OpenAI, Anthropic, Gemini).
- [x] Python AST syntax verification guard to protect code block syntax.
- [x] Streamlit dashboard with real-time parameter tuning and Fleet Learning simulation.

### ✅ v1.1.0 — OpenAI SDK 1-Line Drop-in & Production CI (Current)
- [x] 1-Line Transparent OpenAI SDK wrapper: `wrap_openai_client(client)`.
- [x] Precise token accounting via `tiktoken` (`o200k_base` / `cl100k_base`).
- [x] Real-time cost analytics (`est_cost_savings_usd`) based on current OpenAI pricing tiers.
- [x] Multi-Python CI/CD testing matrix (Python 3.10, 3.11, 3.12) with 100% green unit tests.
- [x] Strict test suite isolation and repository hygiene.

### 🎯 v1.2.0 — OpenAI Structured Outputs & Prompt Caching (Q3 2026)
- [ ] **Structured Outputs / JSON Mode Guard**:
  - Contextual compression that automatically detects and preserves JSON Schema keys, enums, and required properties.
- [ ] **OpenAI Prompt Caching Prefix Co-Optimization**:
  - OpenAI offers a 50% discount on cached prompt prefixes (1,024+ tokens).
  - Implement prefix-invariant boundary detection so the compression engine only mutates dynamic suffix content, maximizing both cache hit rate and token pruning.
- [ ] **Batch Processing API**:
  - High-throughput asynchronous batch compression for high-volume enterprise ingestion.

### 🚀 v2.0.0 — Reasoning Models & Distributed Parameter Sync (Q4 2026)
- [ ] **Reasoning Models Adaptation (`o1` / `o3-mini`)**:
  - Contextual policy tuning specifically adapted for reasoning-heavy prompts.
- [ ] **Streaming Compression Middleware**:
  - Chunked prompt preprocessing with zero impact on Time-To-First-Token (TTFT).
- [ ] **Federated Bandit Updates**:
  - Encrypted, privacy-preserving parameter updates across distributed edge workers.

---

## 💡 How to Propose New Milestones
If you have suggestions or need custom provider support, please open an issue or submit a feature request on [GitHub Issues](https://github.com/howwang0507/Adaptive-Prompt-Compressor/issues).
