# Adaptive Prompt Compressor 🧠📉

**Dynamic LLM context optimization using Contextual Multi-Armed Bandits (LinUCB).**  
Achieve **93.5% reliability** while reducing token costs by dynamically routing prompts through task-aware compression strategies. Optimized for real-time inference with **< 1ms latency**.

---

📄 **[Read the Full Paper (LaTeX/PDF) in `./latex/main.tex`](./latex/main.tex)** | 🚀 **[Try the Interactive Demo (Streamlit)](#-interactive-demo)**

---

## 🌟 Key Features & Academic Highlights

1. **Hybrid Neural-Structural Context ($R^{12}$)**: Unlike static methods, our system uses an expanded 12-dimensional feature vector. It integrates **SBERT-derived neural embeddings** for deep semantic understanding with traditional structural metrics (Length, Diversity, Codeness).
2. **Multi-Provider Support**: Production-ready environments for **Google Gemini, OpenAI (GPT-4o), and Anthropic (Claude 3.5)**.
3. **AST-based Hard Metrics**: For technical tasks, the system incorporates real-time **Syntax Validation** to ensure compressed code remains executable.
4. **Ultra-Low Latency ($O(d^2)$)**: The LinUCB algorithm guarantees a computational complexity of $O(d^2)$. Routing overhead is strictly **< 1ms**, ideal for real-time asynchronous pipelines.
5. **Distributed Fleet Learning**: Decoupled state management using **Redis** enables asynchronous weight synchronization across heterogeneous worker clusters.
6. **Reliability-First Emergence**: In high-penalty environments, the agent autonomously learns to protect structural logic, achieving a **93.5% Success Rate**.

## 📊 Performance Summary

| Environment | Token Saved (%) | Success Rate (%) | Semantic Score | Preferred Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Large-Scale Simulation** | 1.4% | **93.5%** | 0.923 | Reliability-First |
| **Code / Technical Logic** | 2.1% | 95.0% | 0.941 | Arm 0 (Conservative) |
| **Chat / Summarization** | 42.5% | 92.0% | 0.918 | Arm 2 (Aggressive) |

## 🚀 Quick Start (Installation & Usage)

### Option 1: Quick Deployment via Docker Compose 🐳 (Recommended)
If you want to run the Interactive Dashboard with a Redis Parameter Server instantly:

```bash
# 1. Clone the repository
git clone https://github.com/howwang0507/Adaptive-Prompt-Compressor.git
cd Adaptive-Prompt-Compressor

# 2. Setup environment variables
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY, OPENAI_API_KEY, etc.

# 3. Start the application stack
docker-compose up -d

# 4. Access the UI
# Open your browser and navigate to http://localhost:8501
```

### Option 2: Local Development Setup (using `uv`)

```bash
# 1. Clone the repository
git clone https://github.com/howwang0507/Adaptive-Prompt-Compressor.git
cd Adaptive-Prompt-Compressor

# 2. Install dependencies using uv
uv sync

# 3. Setup environment variables
cp .env.example .env

# 4. Run the Streamlit Demo locally
uv run streamlit run src/app.py
```

### 💻 Basic Usage (Code Integration)

Integrate the adaptive compressor into your Python project or MCP Server in just a few lines. See `examples/basic_usage.py` for more details.

```python
from src.interface import LinUCBCompressor
import os

# Initialize (defaults to Simulation mode if no key provided)
compressor = LinUCBCompressor(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-1.5-flash")

# Compress your prompt
prompt = "def calculate_fibonacci(n): ..."
compressed_text, strategy, metadata = compressor.compress(prompt)

print(f"Strategy: {strategy} | Compressed: {compressed_text}")
```

**2. Secret Management**
Create a `.env` file from the template:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run via Docker
```bash
docker build -t prompt-compressor .
docker run -p 8501:8501 prompt-compressor
```

## 📁 Repository Structure

```text
Adaptive-Prompt-Compressor/
├── src/                    # Core Architecture & App
│   ├── app.py              # Streamlit Interactive Dashboard
│   ├── agent.py            # LinUCB CMAB Implementation
│   ├── environment.py      # Multi-provider Env (Simulation & API)
│   └── utils.py            # Reward functions & Semantic metrics
├── scripts/                # Sim2Real Reproduction Scripts
├── latex/                  # Publication-ready Manuscript
├── Dockerfile              # Containerized Environment
├── pyproject.toml          # Modern dependency management (uv)
└── CITATION.cff            # Academic citation metadata
```

## ⚡ High-Throughput & Telemetry (Production-Ready)

Designed for high-performance middleware requirements:

- **Asynchronous Batching**: Built-in `asyncio` support for concurrent prompt processing.
- **Real-time Telemetry (SSE)**: Stream routing decisions, latency, and rewards to your monitoring dashboard via Server-Sent Events.
- **Auto-Fallback (Reliability)**: Automatically retries with original prompts if semantic fidelity drops below a defined threshold (default: 0.6).

```python
from src.async_interface import AsyncLinUCBCompressor

# Initialize async compressor with quality threshold
compressor = AsyncLinUCBCompressor(fallback_threshold=0.8)

# Parallel batch processing
results = await compressor.compress_batch([
    "Prompt 1...", "Prompt 2...", "Prompt 3..."
])
```

## 📡 Observability

Monitor your bandit's performance in real-time using our SSE telemetry server:

```bash
uv run python src/telemetry.py
```

## 🗄️ Persistence & Analytics

The system now utilizes a **SQLite-backed database** (`results/experiments.db`) for robust persistence and SQL-based analytics.

- **Complex Queries**: Use SQL `HAVING` clauses to filter performance by category density.
- **Scalability**: Designed to handle 100k+ trials with indexed search.

---

## 🧠 Advanced Algorithmic Optimizations

To ensure production stability and academic rigor, our LinUCB implementation includes several state-of-the-art features:

1. **Online Feature Scaling**: Uses **Welford's Algorithm** to dynamically normalize features (e.g., scaling character count down to match binary codeness flags). This prevents large-scale features from dominating the covariance matrix.
2. **Concept Drift Adaptation**: Implements a **Forgetting Factor ($\gamma=0.99$)** to allow the agent to "unlearn" stale data. This is critical for adapting to silent LLM model updates or shifts in user prompt distributions.
3. **Dual-Track Reward Mechanism**: 
   - **Online Track**: Lightweight heuristics (keyword retention, length ratio) for zero-latency feedback.
   - **Offline Track**: Deep semantic evaluation (BERTScore/LLM-as-a-judge) for policy calibration.
4. **Explainable AI (XAI)**: The agent's "thought process" is fully transparent. You can visualize the learned $\theta$ weights to see exactly why the agent avoids aggressive compression for code-heavy prompts.

---

## 🔍 Explainability & Weight Visualization

Visualize what the agent has learned:

```bash
uv run python scripts/visualize_weights.py
```
*(Produces a heatmap in `assets/figure_4_weights.png` showing feature-to-strategy correlations).*

---

## 🛰️ Research & Edge Computing Use-Cases

The **Adaptive Prompt Compressor** is uniquely positioned for systems where bandwidth is expensive and reliability is non-negotiable:

- **Robotic Edge Intelligence**: Autonomous robots (UAVs/AMRs) translating raw sensor data into LLM prompts. Our system acts as the "Prefrontal Cortex," ensuring critical spatial logic is preserved while minimizing transmission latency to the cloud.
- **Real-time System Monitoring**: Processing million-line logs into diagnostic summaries. The **Reliability-First** policy prevents the accidental removal of rare error codes or negations in SQL queries.
- **Agentic Interactions**: High-frequency multi-agent communication where every token saved extends the operation window under API rate limits.

---

## 🛡️ Production Stability & Flawless Engineering

This project is built for mission-critical LLM deployments, featuring 'Temple-Level' stability optimizations:

1. **Numerical Stability ($O(d^2)$ SM Update)**: Instead of costly and unstable $O(d^3)$ matrix inversions, we use the **Sherman-Morrison formula** for incremental updates. This prevents floating-point drift and guarantees invertible covariance matrices through **Ridge Regularization**.
2. **Thread-Safe Architecture**: All matrix operations and agent updates are protected by **Atomic Locks**, ensuring the compressor can be safely deployed in high-concurrency environments like FastAPI or asynchronous workers.
3. **OOD Input Protection**: A built-in **Feature Guard** monitors real-time input distributions. If a prompt's features are Out-of-Distribution (OOD), the system automatically triggers a **Conservative Fallback** to protect the inference pipeline from radical bandit decisions.
4. **Type-Safe Discipline**: 100% code coverage with **Python Type Hints**, validated by `mypy` and `ruff`.

---

## 🏛️ Enterprise & Security Architecture (Roadmap)

For distributed, massive-scale deployments (e.g., K8s clusters serving millions of daily requests), the architecture is designed to support the following ceiling-level extensions:

- **Distributed Parameter Synchronization**: Transitioning from in-memory locks to a **Redis Parameter Server**. This allows hundreds of stateless worker pods to asynchronously push matrix incremental updates (using Redis atomic transactions) ensuring global fleet learning without locking bottlenecks.
- **Adversarial Resilience (Guardrail Masking)**: LLM Prompt Injection attacks often try to overwrite system prompts. The compressor is designed to accept `System Instructions` that bypass the compression heuristic (Freeze Masking), ensuring that safety guardrails are never stripped out to save tokens.
- **Zero-Shot Domain Adaptation (Meta-Learning)**: The `LinUCB` initialization supports injecting pre-trained `domain_priors`. For a new customer deploying in a specific domain (e.g., Medical Triage), the agent can be warm-started with pre-calculated $\theta$ matrices, avoiding the cold-start exploration penalty.
- **Attention Sink Protection**: Integrating a "Position Sensitivity" feature dimension to protect the boundaries (Start/End) of the prompt, respecting the "Lost in the Middle" phenomenon observed in Deep Transformer architectures.

---

## 🎓 Citation

```bibtex

@article{Wang2026Adaptive,
  title={Adaptive Prompt Compression via Contextual Bandits: Balancing Token Cost and Semantic Fidelity in Resource-Constrained Environments},
  author={MINGHAO WANG},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/howwang0507/Adaptive-Prompt-Compressor}
}
```

---
*Developed for robust, enterprise-grade LLM inference optimization.*
