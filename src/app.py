import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import SimulatedEnvironment, RealLLMEnvironment
from src.utils import calculate_reward

st.set_page_config(
    page_title="Adaptive Prompt Compressor Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Adaptive Prompt Compressor")
st.markdown("""
### Contextual Bandits for Dynamic LLM Routing
This interactive dashboard demonstrates how a **LinUCB Contextual Bandit** learns to optimize prompt compression strategies based on linguistic features and real-time feedback.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.selectbox("Execution Mode", ["Simulation (Offline)", "Live API (GCP)"])
alpha = st.sidebar.slider("Exploration Factor (Alpha)", 0.1, 2.0, 1.0)

if mode == "Live API (GCP)":
    api_key = st.sidebar.text_input("GCP API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        st.warning("Please provide a Gemini API Key to run live tests.")
    env = RealLLMEnvironment(api_key) if api_key else None
else:
    env = SimulatedEnvironment()

# Initialize Agent
if 'agent' not in st.session_state:
    st.session_state.agent = LinUCB(n_arms=3, n_features=5, alpha=alpha)
agent = st.session_state.agent

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Prompt")
    user_input = st.text_area(
        "Enter a prompt to compress:",
        "Write a robust Python function that uses a thread-safe dictionary to count word frequencies in a large text file, handling FileNotFoundError and other potential exceptions.",
        height=150
    )
    
    if st.button("🚀 Process with LinUCB"):
        if mode == "Live API (GCP)" and env is None:
            st.error("API Key required for Live Mode.")
        else:
            # 1. Feature Extraction
            features = env.extract_features(user_input)
            
            # 2. Agent Decision
            # Calculate UCB scores for visualization
            p = []
            for a in range(agent.n_arms):
                x = features.reshape(-1, 1)
                theta = agent.A_inv[a] @ agent.b[a]
                ucb = theta.T @ x + agent.alpha * np.sqrt(x.T @ agent.A_inv[a] @ x)
                p.append(ucb.item())
            
            arm = agent.select_arm(features)
            
            # 3. Execution
            with st.spinner(f"Agent selected Arm {arm}... Running..."):
                res = env.execute_request(user_input, arm)
                
            # 4. Display Results
            st.success(f"Strategy Selected: **Arm {arm}** ({['Conservative', 'Moderate', 'Aggressive'][arm]})")
            
            st.info(f"**Compressed Prompt:**\n\n{res['compressed_prompt']}")
            
            # Feature display
            st.write("---")
            st.write("🔍 **Extracted Context Vector ($s_t$):**")
            feat_df = pd.DataFrame({
                "Feature": ["Length", "Lexical Diversity", "Codeness", "Complexity", "Bias"],
                "Value": features
            })
            st.table(feat_df)

with col2:
    st.subheader("📈 Decision Analytics")
    
    if 'features' in locals():
        # UCB Scores Bar Chart
        fig_ucb = go.Figure(data=[
            go.Bar(name='UCB Score', x=['Arm 0 (Cons)', 'Arm 1 (Mod)', 'Arm 2 (Aggr)'], y=p, marker_color=['#636EFA', '#EF553B', '#00CC96'])
        ])
        fig_ucb.update_layout(title="Agent Confidence (UCB Scores) per Arm", yaxis_title="Score")
        st.plotly_chart(fig_ucb, use_container_width=True)
        
        # Token Savings Gauges
        saving_pct = (1 - (res['comp_tokens'] / res['base_tokens'])) * 100
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = saving_pct,
            title = {'text': "Token Savings (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "darkblue"},
                     'steps' : [
                         {'range': [0, 20], 'color': "lightgray"},
                         {'range': [20, 50], 'color': "gray"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Historical Results ---
st.write("---")
st.subheader("📜 Recent Experiment Logs")

# Load latest results
result_files = [f for f in os.listdir("results") if f.endswith(".csv")]
if result_files:
    latest_file = max([os.path.join("results", f) for f in result_files], key=os.path.getctime)
    df_logs = pd.read_csv(latest_file)
    st.dataframe(df_logs.tail(10), use_container_width=True)
    
    # Cumulative Reward Plot
    if 'reward' in df_logs.columns:
        fig_reward = px.line(df_logs, x=df_logs.index, y=df_logs['reward'].expanding().mean(), 
                             title="Cumulative Average Reward (Learning Curve)")
        st.plotly_chart(fig_reward, use_container_width=True)
else:
    st.write("No logs found. Run a benchmark to see history.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Wang Hao**")
st.sidebar.write("Project: Adaptive Prompt Compressor")
