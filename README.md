# Adaptive Prompt Compression via Contextual Bandits 🧠📉

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Gemini API](https://img.shields.io/badge/Model-Gemini%201.5%20Flash-orange.svg)

> **Balancing Token Cost and Semantic Fidelity in Resource-Constrained Environments**

This repository contains the official implementation of our research on applying **Contextual Multi-Armed Bandits (LinUCB)** to dynamic LLM prompt compression. By learning from real-time execution feedback, our agent autonomously balances token economy against the high penalty of semantic failures, resulting in an emergent **Reliability-First** routing policy.

📄 **[Read the Full Paper (LaTeX/PDF) in `./latex/main.tex`](./latex/main.tex)**

---

## 🌟 Key Features & Academic Highlights

1. **Adaptive Contextual Routing**: Unlike static compression methods (e.g., LLMLingua) that rely on perplexity, our system uses a $d=5$ feature vector (Length, Lexical Diversity, Structural Codeness, Semantic Entropy, Bias) to dynamically select compression strategies.
2. **Ultra-Low Latency ($O(d^2)$)**: The LinUCB algorithm mathematically guarantees a computational complexity of $O(d^2)$. For our 5-dimensional feature space, the routing overhead is strictly **< 1ms**, making it fully viable for real-time, asynchronous LLM pipelines (e.g., SSE streams).
3. **Sim2Real Transferability**: Extensively validated across both a 4,500-step Offline Simulation and Live API Deployment (Gemini 1.5 Flash).
4. **Reliability-First Emergence**: In high-penalty production environments ($\lambda_f = 2.5$), the agent autonomously learns to protect structural logic (e.g., Python code, negations), sacrificing marginal token savings to achieve a dominant **93.5% Success Rate**.

## 📊 Performance Summary

| Environment | Token Saved (%) | Success Rate (%) | Semantic Score | Preferred Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Large-Scale Simulation** | 1.4% | **93.5%** | 0.923 | Reliability-First |
| **Code / Technical Logic** | 2.1% | 95.0% | 0.941 | Arm 0 (Conservative) |
| **Chat / Summarization** | 42.5% | 92.0% | 0.918 | Arm 2 (Aggressive) |

*(Note: Feature Ablation studies confirm that removing the "Codeness" regex feature drops the success rate catastrophically to 71.0%).*

## 📁 Repository Structure

```text
Adaptive-Prompt-Compressor/
├── src/                    # Core Architecture
│   ├── agent.py            # LinUCB CMAB Implementation
│   ├── environment.py      # Multi-provider Env (Simulation & Real API)
│   └── utils.py            # Reward functions & Semantic metrics
├── scripts/                # Sim2Real Reproduction Scripts
│   ├── run_benchmarks.py   # Large-scale simulation (n=1500)
│   └── mass_real_api_test.py # Live API validation against Vertex/AI Studio
├── latex/                  # Publication-ready Manuscript
│   └── main.tex            # NeurIPS 2025 format paper
├── assets/                 # Generated Figures (Learning Curve, Ablation)
└── results/                # CSV logs from all experiments
```

## 🚀 Quick Start (Reproduction)

**1. Setup Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Run Offline Simulation**
```bash
python scripts/run_benchmarks.py
```

**3. Run Live API Test (Sim2Real)**
```bash
export GEMINI_API_KEY="your_api_key_here"
python scripts/mass_real_api_test.py
```

## 🎓 Citation

If you find this codebase or methodology useful in your research, please consider citing:

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
