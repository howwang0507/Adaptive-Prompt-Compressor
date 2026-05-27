# Adaptive Prompt Compression via Contextual Bandits 🧠📉

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Gemini API](https://img.shields.io/badge/Model-Gemini%201.5%20Flash-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)

> **Balancing Token Cost and Semantic Fidelity in Resource-Constrained Environments**

This repository contains the official implementation of our research on applying **Contextual Multi-Armed Bandits (LinUCB)** to dynamic LLM prompt compression. By learning from real-time execution feedback, our agent autonomously balances token economy against the high penalty of semantic failures, resulting in an emergent **Reliability-First** routing policy.

📄 **[Read the Full Paper (LaTeX/PDF) in `./latex/main.tex`](./latex/main.tex)** | 🚀 **[Try the Interactive Demo (Streamlit)](#-interactive-demo)**

---

## 🌟 Key Features & Academic Highlights

1. **Adaptive Contextual Routing**: Unlike static compression methods (e.g., LLMLingua), our system uses a $d=5$ feature vector (Length, Lexical Diversity, Structural Codeness, Semantic Entropy, Bias) to dynamically select compression strategies.
2. **Ultra-Low Latency ($O(d^2)$)**: The LinUCB algorithm guarantees a computational complexity of $O(d^2)$. Routing overhead is strictly **< 1ms**, ideal for real-time asynchronous pipelines.
3. **Sim2Real Transferability**: Validated across a 4,500-step Offline Simulation and Live API Deployment (Gemini 1.5 Flash).
4. **Reliability-First Emergence**: In high-penalty environments ($\lambda_f = 2.5$), the agent autonomously learns to protect structural logic, achieving a **93.5% Success Rate**.

## 📊 Performance Summary

| Environment | Token Saved (%) | Success Rate (%) | Semantic Score | Preferred Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Large-Scale Simulation** | 1.4% | **93.5%** | 0.923 | Reliability-First |
| **Code / Technical Logic** | 2.1% | 95.0% | 0.941 | Arm 0 (Conservative) |
| **Chat / Summarization** | 42.5% | 92.0% | 0.918 | Arm 2 (Aggressive) |

## 🚀 Quick Start (Usage)

Integrate the adaptive compressor into your Python project in just 3 lines:

```python
from src.interface import LinUCBCompressor

# Initialize (defaults to Simulation mode if no key provided)
compressor = LinUCBCompressor(api_key="your_api_key", model_name="gemini-1.5-flash")

# Compress your prompt
prompt = "Write a complex Python function to handle thread-safe file operations..."
compressed_text, strategy, metadata = compressor.compress(prompt)

print(f"Strategy: {strategy} | Compressed: {compressed_text}")
```

## 🛠️ Setup & Environment

**1. Dependency Management (using uv)**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Sync dependencies
uv sync
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

## 🎓 Citation

If you use this research, please click "Cite this repository" on the sidebar or use the following:

```bibtex
@article{Wang2026Adaptive,
  title={Adaptive Prompt Compression via Contextual Bandits: Balancing Token Cost and Semantic Fidelity in Resource-Constrained Environments},
  author={Hao, Wang},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/howwang0507/Adaptive-Prompt-Compressor}
}
```

---
*Developed for robust, enterprise-grade LLM inference optimization.*
