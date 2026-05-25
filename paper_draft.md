# Adaptive Prompt Compression via Contextual Bandits (LinUCB)

## Abstract
With the proliferation of Large Language Models (LLMs), optimizing API costs while maintaining generation quality has become a critical challenge. This paper presents an adaptive prompt compression framework using the **LinUCB Contextual Bandit** algorithm. By dynamically selecting compression strategies based on linguistic features, the system balances token savings with semantic integrity. Experimental results demonstrate that our approach achieves significant cost reduction while maintaining high response validity, even under extreme resource constraints.

## 1. Introduction
### 1.1 Motivation
LLM pricing models are typically token-based and limited by context window sizes. Existing compression methods often rely on static rules, failing to account for the sensitivity of different content types. For instance, whitespace removal might break Python code logic, while stopword filtering is highly effective for casual conversations.

### 1.2 Objective
Develop an intelligent routing system that dynamically navigates the trade-off between **Cost (Tokens)** and **Quality (Validity)**.

## 2. Methodology
### 2.1 Feature Extraction
Input text $x$ is mapped to a feature vector $f(x)$, including:
- Normalized text length.
- Unique word ratio (lexical diversity).
- Structural features (presence of code keywords like `def`, `{`).
- Average word length.

### 2.2 Action Space (Arms)
The agent chooses between three strategies (Arms):
- $a_0$ (Raw): No processing (High quality, high cost).
- $a_1$ (Basic): Strips extra whitespaces and newlines.
- $a_2$ (Aggressive): Filters stopwords, retaining only semantic entities (Low cost, higher risk).

### 2.3 Core Algorithm: LinUCB
We utilize the LinUCB algorithm to handle the **Exploration vs. Exploitation** trade-off. For each arm $a$, the estimated upper confidence bound $p_{t,a}$ is:
$$p_{t,a} \stackrel{\text{def}}{=} \theta_a^\top x_{t,a} + \alpha \sqrt{x_{t,a}^\top A_a^{-1} x_{t,a}}$$
Where:
- $A_a$ tracks the feature covariance.
- $\theta_a$ represents the weight parameters for Arm $a$.
- $\alpha$ is the exploration coefficient.

### 2.4 Reward Function Design
The reward function is designed to penalize failures heavily:
$$Reward = w_1 \cdot \text{Saving} - w_2 \cdot \text{Latency} - w_3 \cdot \text{FailurePenalty}$$
We set $w_1=1.5, w_2=0.2, w_3=2.5$ to ensure the agent avoids strategies that trigger blocked or invalid LLM responses.

## 3. Experimental Setup
- **Model**: Google Gemini 1.5 Flash (via API).
- **Interface**: Interactive Streamlit Dashboard.
- **Constraints**: Evaluated under extreme daily API quotas (20 requests/day) to test **Sample Efficiency**.

## 4. Results and Discussion
### 4.1 Sample Efficiency under Quota Constraints
Experimental data shows that despite strict API limits, LinUCB successfully identifies successful signals (e.g., Step 18 in current logs). The negative reward feedback correctly steers the agent away from risky compression on sensitive data.

### 4.2 Robustness to Code Sensitivity
The agent learns to assign $a_0$ or $a_1$ to snippets identified as "Code," while aggressively applying $a_2$ to "Chat" categories, optimizing both cost and functionality.

## 5. Conclusion
This study demonstrates the feasibility of using reinforcement learning for automated prompt management. Future work includes exploring multi-model routing and offline pre-training on large-scale datasets.
