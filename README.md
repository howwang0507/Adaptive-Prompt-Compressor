# 🧠 Adaptive Prompt Compression via Contextual Bandits

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adaptive-prompt-compreappr-wanghao.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Optimizing LLM efficiency where every token and every request counts. Learning adaptive strategies under extreme resource constraints."**

## 📖 Abstract
Large Language Models (LLMs) incur significant operational costs and latency due to token-based pricing. While static prompt compression methods exist, they often fail to adapt to the semantic sensitivity of diverse tasks. 

This repository implements a novel **Adaptive Prompt Compression** framework using **LinUCB Contextual Bandits**. The system dynamically routes prompts through various compression strategies based on real-time feedback, achieving a **16.0% reduction in token usage** while maintaining **88.0% response validity** in simulated environments.

## ✨ Key Features
* **Adaptive Routing**: Automatically switches between Raw, Basic, and Aggressive compression based on linguistic features (e.g., text length, lexical diversity, "codeness").
* **Resource-Constrained Learning**: Optimized for strict API quotas (e.g., Free-tier 20 RPM limits), reaching stable policy convergence in under 50 steps.
* **Multi-Objective Reward**: Balances cost-savings, latency, and semantic fidelity.
* **Sim2Real Environment**: Includes a comprehensive offline simulation mode alongside the live Gemini API environment.

## 🏗️ System Architecture
The agent frames prompt optimization as a **Contextual Multi-Armed Bandit (CMAB)** problem:
1. **Feature Extractor**: Analyzes the incoming prompt ($x_t$).
2. **LinUCB Agent**: Evaluates the context and selects an action ($a_t$) to maximize the expected reward.
3. **Compression Arms**: 
   * `Arm 0 (Raw)`: No compression (Quality ceiling).
   * `Arm 1 (Basic)`: Whitespace & newline stripping.
   * `Arm 2 (Aggressive)`: Stopword filtration.

## 📊 Experimental Results
Our agent significantly outperforms static rule-based baselines, successfully learning a conservative policy for sensitive data (Code) while maximizing economic efficiency for redundant data (Chat).

| Method | Avg. Reward | Token Saved (%) | Success Rate (%) |
| :--- | :---: | :---: | :---: |
| Raw | -0.29 | 0.0% | 98.0% |
| Static Rule | -0.33 | 8.2% | 85.0% |
| **LinUCB (Ours)** | **-0.18** | **16.0%** | **88.0%** |

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/howwang0507/Adaptive-Prompt-Compressor.git
cd Adaptive-Prompt-Compressor
pip install -r requirements.txt
```

### 2. Run the Dashboard
Launch the interactive Streamlit experiment dashboard:
```bash
./run.sh
# Or run directly: streamlit run app.py
```

### 3. Usage Modes
* **Offline Simulation**: Test the algorithm's convergence without needing an API key.
* **Real API (Gemini)**: Input your Google Gemini API key to test real-world token savings and response validity.

## 📂 Repository Structure
- `src/agent.py`: Implementation of the **LinUCB** algorithm.
- `src/environment.py`: Simulated and Real LLM environment logic.
- `src/utils.py`: Multi-objective reward calculation and feature extraction.
- `app.py`: Streamlit UI for experiment tracking and visualization.
- `paper_draft.md`: Full academic manuscript detailing the methodology.

## 📜 Citation
If you find this project useful for your research, please refer to the `paper_draft.md` for full methodological details. (BibTeX citation coming soon).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
