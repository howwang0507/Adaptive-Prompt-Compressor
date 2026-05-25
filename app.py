import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from src.agent import LinUCB
from src.environment import RealLLMEnvironment, SimulatedEnvironment
from src.utils import calculate_reward, get_test_dataset

# ==========================================
# 🌍 全局設定與資料集
# ==========================================
st.set_page_config(page_title="Adaptive LLM Optimizer V2", layout="wide", page_icon="🧠")

REAL_DATA = [
    # --- Chat Category ---
    {"text": "Could you please explain to me in detail what the weather is usually like in Tokyo during the spring season?", "category": "Chat"},
    {"text": "Tell me about the history of the Eiffel Tower, including when it was built and who designed it.", "category": "Chat"},
    {"text": "What are some fun things to do in San Francisco for a first-time visitor?", "category": "Chat"},
    {"text": "Can you give me a list of healthy snacks that are easy to prepare at home?", "category": "Chat"},
    {"text": "I'm feeling a bit stressed today. Could you suggest some relaxation techniques?", "category": "Chat"},
    
    # --- Code Category ---
    {"text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)", "category": "Code"},
    {"text": "Write a Python function to scrape news headlines from a website using BeautifulSoup.", "category": "Code"},
    {"text": "const factorial = (n) => n === 0 ? 1 : n * factorial(n - 1);", "category": "Code"},
    
    # --- QA Category ---
    {"text": "Who was the first person to walk on the moon, and what was the name of the mission?", "category": "QA"},
    {"text": "What are the primary causes of climate change, and how do greenhouse gases contribute?", "category": "QA"},
    {"text": "Who wrote the play 'Romeo and Juliet' and in what century was it written?", "category": "QA"},

    # --- Summarization Category ---
    {"text": "The Industrial Revolution was a period of global economic transition towards more efficient manufacturing processes. Summarize this passage in one sentence.", "category": "Summarization"},
    {"text": "A supernova is a powerful and luminous stellar explosion occurring during the last evolutionary stages of a star. Provide a brief summary.", "category": "Summarization"},

    # --- Translation Category ---
    {"text": "Translate the following English sentence into traditional Chinese: 'The quick brown fox jumps over the lazy dog.'", "category": "Translation"},
    {"text": "How do you say 'Where is the nearest library?' in Spanish, French, and German?", "category": "Translation"}
]

def run_experiment(dataset, mode, agent, env):
    logs = []
    cumulative_reward = 0
    for i, data in enumerate(dataset):
        features = env.extract_features(data["text"])
        
        if mode == "Baseline": arm = 0
        elif mode == "Rule_Based": 
            arm = 0 if features[2] == 1.0 else (2 if len(data["text"]) > 100 else 1)
        elif mode == "LinUCB": 
            arm = agent.select_arm(features, step=i)
            
        res = env.execute_request(data["text"], arm)
        reward, saving, lat_pen, fail_pen = calculate_reward(res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"])
        
        if mode == "LinUCB": agent.update(arm, features, reward)
            
        cumulative_reward += reward
        logs.append({
            "step": i, "mode": mode, "category": data["category"], "arm": arm, 
            "reward": reward, "cumulative_reward": cumulative_reward, 
            "avg_reward": cumulative_reward / (i+1),
            "saving_ratio": saving, "valid": res["valid"], "llm_response": res["answer"]
        })
    return logs

# ==========================================
# 🌐 UI 介面
# ==========================================
st.title("🧠 基於 Contextual Bandit 之自適應 Prompt 壓縮系統")
st.markdown("### Adaptive Prompt Compression via LinUCB: Balancing Cost and Quality")

with st.expander("📖 研究背景與方法論 (Methodology)", expanded=False):
    st.write("""
    **核心創新：自適應路由 (Adaptive Routing)**
    本研究不使用固定的壓縮規則，而是透過 **Contextual Bandit (LinUCB)** 演算法動態選擇最佳的壓縮策略。
    
    - **Arm 0 (Raw)**: 原文發送。
    - **Arm 1 (Basic)**: 移除多餘空格。
    - **Arm 2 (Aggressive)**: 移除停用詞。
    """)

if "df_logs" not in st.session_state: st.session_state.df_logs = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ 實驗參數設定")
    env_mode = st.radio("環境模式", ["離線模擬 (Simulation)", "真實 API (Real API)"])
    
    api_key = ""
    if env_mode == "真實 API (Real API)":
        api_key = st.text_input("Gemini API Key", type="password")
        st.warning("注意：免費版 API 有 15 RPM 限制。")
    
    st.markdown("---")
    st.subheader("🛠️ 數據源設定")
    data_source = st.radio("選擇數據來源", ["內建資料集", "自定義輸入 Prompt"])
    
    custom_prompt = ""
    if data_source == "自定義輸入 Prompt":
        custom_prompt = st.text_area("輸入自定義文本", height=150)
    
    st.subheader("🧪 實驗模式選擇")
    selected_modes = st.multiselect("選擇要執行的模式", ["Baseline", "Rule_Based", "LinUCB"], default=["LinUCB"])
    
    if st.button("🚀 開始實驗演示", use_container_width=True, type="primary"):
        if env_mode == "真實 API (Real API)" and not api_key:
            st.error("請輸入 API Key！")
        else:
            all_logs = []
            env = RealLLMEnvironment(api_key) if env_mode == "真實 API (Real API)" else SimulatedEnvironment()
            
            if data_source == "自定義輸入 Prompt" and custom_prompt:
                test_data = [{"text": custom_prompt, "category": "Custom"}] * 50
            else:
                test_data = get_test_dataset(REAL_DATA, 50)
            
            with st.spinner(f"🤖 正在執行 {len(test_data) * len(selected_modes)} 步實驗..."):
                for mode in selected_modes:
                    agent = LinUCB(alpha=1.0) if mode == "LinUCB" else None
                    all_logs.extend(run_experiment(test_data, mode, agent, env))
                st.session_state.df_logs = pd.DataFrame(all_logs)
            st.success("實驗完成！")

if not st.session_state.df_logs.empty:
    df = st.session_state.df_logs
    tab1, tab2, tab3 = st.tabs(["📊 效能分析", "🧠 決策熱圖", "📄 完整日誌"])
    
    with tab1:
        exp_table = df.groupby("mode").agg(
            Reward=("reward", "mean"),
            Saving=("saving_ratio", lambda x: f"{x.mean()*100:.1f}%"),
            Success=("valid", lambda x: f"{x.mean()*100:.1f}%")
        ).reset_index()
        st.table(exp_table)
        
        reward_chart = alt.Chart(df).mark_line().encode(
            x='step', y='avg_reward', color='mode'
        ).properties(height=350).interactive()
        st.altair_chart(reward_chart, use_container_width=True)

    with tab2:
        pivot_df = df[df["mode"]=="LinUCB"].groupby(["category", "arm"]).size().reset_index(name='counts')
        pivot_df['Selection Rate (%)'] = pivot_df.groupby('category')['counts'].transform(lambda x: (x / x.sum() * 100))
        selection_chart = alt.Chart(pivot_df).mark_bar().encode(
            x='category:N', y='Selection Rate (%):Q', color='arm:N'
        ).properties(height=350)
        st.altair_chart(selection_chart, use_container_width=True)

    with tab3:
        st.dataframe(df, use_container_width=True)
