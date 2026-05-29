import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import (
    SimulatedEnvironment,
    RealLLMEnvironment,
    OpenAIEnvironment,
    AnthropicEnvironment,
)
from src.utils import calculate_reward

st.set_page_config(
    page_title="Adaptive Prompt Compressor Demo", page_icon="🧠", layout="wide"
)

st.title("🧠 Adaptive Prompt Compressor")
st.markdown("""
### Neural Contextual Bandits for Dynamic LLM Routing
This interactive dashboard demonstrates an upgraded **Hybrid Neural-LinUCB Bandit**. It combines structural features with SBERT-derived neural embeddings to achieve high-precision prompt routing.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.selectbox(
    "Execution Mode",
    [
        "Simulation (Offline)",
        "Live API (Gemini)",
        "Live API (OpenAI)",
        "Live API (Anthropic)",
    ],
)
alpha = st.sidebar.slider("Exploration Factor (Alpha)", 0.1, 2.0, 1.0)

# Initialize Agent (Upgraded to 12D Feature Space)
if "agent" not in st.session_state:
    st.session_state.agent = LinUCB(n_arms=3, n_features=12, alpha=alpha)

# Pre-train Option
if st.sidebar.button("🎓 Pre-train Neural Agent (Sim 50 steps)"):
    sim_env = SimulatedEnvironment()
    for _ in range(50):
        prompts = [
            "def solve(): return 42",
            "Explain quantum physics in simple terms.",
            "Write a short story about AI.",
        ]
        txt = random.choice(prompts)
        feats = sim_env.extract_features(txt)
        a = st.session_state.agent.select_arm(feats)
        res = sim_env.execute_request(txt, a)
        rew, _, _, _ = calculate_reward(
            res["base_tokens"],
            res["comp_tokens"],
            res["latency"],
            res["valid"],
            semantic_score=0.9,
        )
        st.session_state.agent.update(a, feats, rew)
    st.sidebar.success("Neural Agent pre-trained!")

env = None
if mode == "Live API (Gemini)":
    api_key = st.sidebar.text_input(
        "Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", "")
    )
    if not api_key:
        st.sidebar.warning("Please provide a Gemini API Key to run live tests.")
    else:
        env = RealLLMEnvironment(api_key)
elif mode == "Live API (OpenAI)":
    api_key = st.sidebar.text_input(
        "OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        st.sidebar.warning("Please provide an OpenAI API Key.")
    else:
        env = OpenAIEnvironment(api_key)
elif mode == "Live API (Anthropic)":
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    if not api_key:
        st.sidebar.warning("Please provide an Anthropic API Key.")
    else:
        env = AnthropicEnvironment(api_key)
else:
    env = SimulatedEnvironment()

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Prompt")
    user_input = st.text_area(
        "Enter a prompt to compress:",
        "Write a robust Python function that uses a thread-safe dictionary to count word frequencies in a large text file, handling FileNotFoundError and other potential exceptions.",
        height=150,
    )

    if st.button("🚀 Process with LinUCB"):
        if mode.startswith("Live") and env is None:
            st.error("API Key required for Live Mode.")
        else:
            # 1. Feature Extraction
            features = env.extract_features(user_input)

            # 2. Agent Decision
            p = []
            for a in range(st.session_state.agent.n_arms):
                x = np.array(features).reshape(-1, 1)
                theta = st.session_state.agent.A_inv[a] @ st.session_state.agent.b[a]
                ucb = theta.T @ x + st.session_state.agent.alpha * np.sqrt(
                    x.T @ st.session_state.agent.A_inv[a] @ x
                )
                p.append(ucb.item())

            arm = st.session_state.agent.select_arm(features)

            # 3. Execution
            with st.spinner(f"Agent selected Arm {arm}... Running..."):
                res = env.execute_request(user_input, arm)
                compressed_text = env.compress_prompt(user_input, arm)

            # 4. Display Results
            st.success(
                f"Strategy Selected: **Arm {arm}** ({['Conservative', 'Moderate', 'Aggressive'][arm]})"
            )
            st.info(f"**Compressed Prompt:**\n\n{compressed_text}")

            if "def" in user_input.lower() and not env.validate_syntax(res["answer"]):
                st.error(
                    "❌ Hard Metric Alert: Syntactically Invalid Code generated after compression."
                )

            # Feature display
            st.write("---")
            st.write("🔍 **Extracted 12D Context Vector (Hybrid Neural):**")
            st.write(features)

with col2:
    st.subheader("📈 Decision Analytics")

    if "features" in locals():
        # UCB Scores Bar Chart
        fig_ucb = go.Figure(
            data=[
                go.Bar(
                    name="UCB Score",
                    x=["Arm 0 (Cons)", "Arm 1 (Mod)", "Arm 2 (Aggr)"],
                    y=p,
                    marker_color=["#636EFA", "#EF553B", "#00CC96"],
                )
            ]
        )
        fig_ucb.update_layout(
            title="Agent Confidence (UCB Scores) per Arm", yaxis_title="Score"
        )
        st.plotly_chart(fig_ucb, use_container_width=True)

        # Token Savings Gauges
        saving_pct = (
            (1 - (res["comp_tokens"] / res["base_tokens"])) * 100
            if res["base_tokens"] > 0
            else 0
        )
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=saving_pct,
                title={"text": "Token Savings (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 20], "color": "lightgray"},
                        {"range": [20, 50], "color": "gray"},
                    ],
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Historical Results & Weights ---
st.write("---")
st.subheader("📜 Analytics & Weights")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown(r"#### Neural Weight Heatmap ($\theta$)")
    weights = st.session_state.agent.get_weights()
    v_min = np.min(weights) if np.any(weights) else -0.1
    v_max = np.max(weights) if np.any(weights) else 0.1

    feat_labels = ["Len", "Div", "Code"] + [f"N_{i}" for i in range(8)] + ["Bias"]

    fig_heatmap = px.imshow(
        weights,
        labels=dict(x="Features", y="Arms", color="Weight"),
        x=feat_labels,
        y=["Arm 0", "Arm 1", "Arm 2"],
        color_continuous_scale="RdBu_r",
        zmin=v_min,
        zmax=v_max,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col4:
    st.markdown("#### Recent Experiment Logs")

# Load latest results
result_files = [f for f in os.listdir("results") if f.endswith(".csv")]
if result_files:
    latest_file = max(
        [os.path.join("results", f) for f in result_files], key=os.path.getctime
    )
    df_logs = pd.read_csv(latest_file)
    st.dataframe(df_logs.tail(10), use_container_width=True)

    # Cumulative Reward Plot
    if "reward" in df_logs.columns:
        fig_reward = px.line(
            df_logs,
            x=df_logs.index,
            y=df_logs["reward"].expanding().mean(),
            title="Cumulative Average Reward (Learning Curve)",
        )
        st.plotly_chart(fig_reward, use_container_width=True)
else:
    st.write("No logs found. Run a benchmark to see history.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **MINGHAO WANG**")
st.sidebar.write("Project: Adaptive Prompt Compressor")
