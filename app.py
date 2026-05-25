import streamlit as st
import numpy as np
import pandas as pd
import random
import altair as alt
import time
import re
import google.generativeai as genai

# ==========================================
# 🧠 核心演算法 (LinUCB)
# ==========================================
class LinUCB:
    def __init__(self, n_arms=3, n_features=5, alpha=1.0):
        self.n_arms, self.n_features, self.alpha = n_arms, n_features, alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
    def select_arm(self, context, step):
        x = np.array(context).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            # UCB formula: mean + alpha * standard_deviation
            # Fix DeprecationWarning by using .item() to get scalar value
            p[a] = (A_inv.dot(self.b[a]).T.dot(x) + self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))).item()
        return np.argmax(p)
        
    def update(self, arm, context, reward):
        x = np.array(context).reshape(-1, 1)
        self.A[arm] += x.dot(x.T)
        self.b[arm] += reward * x

# ==========================================
# 🌍 真實 LLM 環境 (Google Gemini API)
# ==========================================
class RealLLMEnvironment:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # 嘗試使用最新穩定版名稱
        self.model_name = 'gemini-1.5-flash'
        try:
            print("[INFO] Checking available models...")
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            print(f"[INFO] Available models: {available_models}")
            
            # 優先級排序：1.5-flash -> flash-latest -> any flash
            target_models = ['gemini-1.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash', 'gemini-2.5-flash']
            
            selected = None
            for target in target_models:
                if any(target in m for m in available_models):
                    selected = next(m for m in available_models if target in m)
                    break
            
            if not selected:
                # 終極備案：找任何包含 flash 的模型
                flash_models = [m for m in available_models if 'flash' in m.lower()]
                if flash_models:
                    selected = flash_models[0]
            
            if selected:
                self.model_name = selected
                print(f"[INFO] Successfully selected model: {self.model_name}")
            else:
                print("[WARNING] No Flash models found. Using the first available model.")
                if available_models:
                    self.model_name = available_models[0]
                    
        except Exception as e:
            print(f"[WARNING] Could not list models: {e}. Falling back to default.")
            
        self.model = genai.GenerativeModel(self.model_name)
        self.stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "of", "to", "in", "for", "with", "on", "at", "by", "this", "that"}

    def extract_features(self, text):
        length, words = len(text), text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        return [np.clip(length / 1000, 0.0, 1.0), np.clip(unique_ratio, 0.0, 1.0), 
                1.0 if "def " in text or "{" in text else 0.0, 
                np.clip(sum(len(w) for w in words) / max(len(words), 1) / 10.0, 0.0, 1.0), 1.0]

    def compress_prompt(self, text, arm):
        if arm == 0: return text 
        if arm == 1: return re.sub(r'\s+', ' ', text).strip()
        if arm == 2:
            # 優化：保留標點符號以維護 Translation 與 Code 的語義
            words = text.split()
            kept_words = [w for w in words if w.lower() not in self.stop_words]
            return " ".join(kept_words)

    def execute_real_request(self, text, arm):
        try: base_tokens = self.model.count_tokens(text).total_tokens
        except: base_tokens = len(text) // 4
        
        compressed_text = self.compress_prompt(text, arm)
        try: comp_tokens = self.model.count_tokens(compressed_text).total_tokens
        except: comp_tokens = len(compressed_text) // 4
        
        start_time = time.time()
        answer = ""
        is_valid = True
        
        try:
            time.sleep(1.2) # 避開 Rate Limit
            response = self.model.generate_content(
                f"Briefly respond to this: {compressed_text}",
                generation_config={"max_output_tokens": 150}
            )
            try:
                answer = response.text
            except ValueError:
                # 處理可能的內容過濾或空回應
                answer = "Error: Response was blocked or empty."
                is_valid = False
            
            # 強化驗證：關鍵字 + 長度檢查
            confusion_flags = ["i don't understand", "not sure", "unclear", "clarify", "聽不懂", "不清楚", "error", "blocked"]
            if any(flag in answer.lower() for flag in confusion_flags):
                is_valid = False
            
            # 改進長度判定：相容中英文
            # 判斷標準：如果是中文，字元數 > 3；如果是英文，單字數 >= 2
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in answer)
            if has_chinese:
                if len(answer.strip()) < 3: is_valid = False
            else:
                if len(answer.split()) < 2: is_valid = False
            
            if len(answer.strip()) == 0: is_valid = False
                
            # 調試資訊：輸出到終端機以便觀察
            clean_ans = answer[:50].replace('\n', ' ')
            print(f"[DEBUG] Model: {self.model_name} | Arm: {arm} | Valid: {is_valid} | Response: {clean_ans}...")
            
        except Exception as e:
            answer = f"API Error: {str(e)}"
            is_valid = False
            print(f"[ERROR] API Call Failed: {str(e)}")
            
        return {"base_tokens": base_tokens, "comp_tokens": comp_tokens, 
                "latency": (time.time() - start_time) * 1000, "valid": is_valid, "answer": answer}

# ==========================================
# 📊 實驗控制器 (Continuous Online Learning)
# ==========================================
def calculate_reward(base_tokens, comp_tokens, latency, valid):
    saving = (base_tokens - comp_tokens) / max(base_tokens, 1)
    # 優化權重：顯著懲罰失敗 (-2.5)，適度懲罰延遲 (-0.2)，強調節省 (1.5x)
    latency_pen = 0.2 * (latency / 1000.0) 
    fail_pen = 2.5 if not valid else 0.0
    reward = (1.5 * saving - latency_pen - fail_pen)
    return reward, saving, latency_pen, fail_pen

def run_experiment(dataset, mode, agent, env):
    logs = []
    cumulative_reward = 0
    for i, data in enumerate(dataset):
        features = env.extract_features(data["text"])
        
        if mode == "Baseline": arm = 0
        elif mode == "Rule_Based": 
            # 靜態規則：代碼不壓，其餘根據長度判斷
            arm = 0 if features[2] == 1.0 else (2 if len(data["text"]) > 100 else 1)
        elif mode == "LinUCB": 
            arm = agent.select_arm(features, step=i)
            
        res = env.execute_real_request(data["text"], arm)
        reward, saving, lat_pen, fail_pen = calculate_reward(res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"])
        
        # 線上學習：在演示過程中全程持續更新，體現 Bandit 特性
        if mode == "LinUCB":
            agent.update(arm, features, reward)
            
        cumulative_reward += reward
        logs.append({
            "step": i, "mode": mode, "category": data["category"], "arm": arm, 
            "reward": reward, "cumulative_reward": cumulative_reward, 
            "avg_reward": cumulative_reward / (i+1),
            "saving_ratio": saving, "valid": res["valid"], "llm_response": res["answer"]
        })
    return logs

# ==========================================
# 🌐 V23 專題正式定稿版 (Research Focused)
# ==========================================
st.set_page_config(page_title="Adaptive LLM Optimizer", layout="wide", page_icon="🧠")

# 研究題目優化
st.title("🧠 基於 Contextual Bandit 之自適應 Prompt 壓縮系統設計")
st.markdown("### Adaptive Prompt Compression via LinUCB: Balancing Cost and Quality")

with st.expander("📖 研究背景與方法論 (Methodology)", expanded=False):
    st.write("""
    **核心創新：自適應路由 (Adaptive Routing)**
    本研究不使用固定的壓縮規則，而是透過 **Contextual Bandit (LinUCB)** 演算法，讓系統學會根據輸入文本的特徵（如：是否為代碼、詞彙多樣性等）動態選擇最佳的壓縮策略。
    
    - **Arm 0 (Raw)**: 原文發送，保證品質但成本最高。
    - **Arm 1 (Basic)**: 移除多餘空格與換行。
    - **Arm 2 (Aggressive)**: 移除停用詞 (Stopwords)，保留標點以維持語義。
    
    **獎勵函數設計 (Optimized Reward Function):**
    $Reward = 1.5 \cdot Saving - 0.2 \cdot Latency - 2.5 \cdot Failure$
    """)

# 實驗資料集 (收斂精簡版 - 30 筆樣本)
REAL_DATA = [
    {"text": "Could you please explain to me in detail what the weather is usually like in Tokyo during the spring season?", "category": "Chat"},
    {"text": "Tell me about the history of the Eiffel Tower, including when it was built and who designed it.", "category": "Chat"},
    {"text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)", "category": "Code"},
    {"text": "Write a Python function to scrape news headlines from a website using BeautifulSoup.", "category": "Code"},
    {"text": "Who was the first person to walk on the moon, and what was the name of the mission?", "category": "QA"},
    {"text": "What are the primary causes of climate change, and how do greenhouse gases contribute?", "category": "QA"},
    {"text": "The Industrial Revolution was a period of global economic transition towards more efficient manufacturing processes. Please summarize this passage in one sentence.", "category": "Summarization"},
    {"text": "A supernova is a powerful and luminous stellar explosion occurring during the last evolutionary stages of a star. Provide a brief summary.", "category": "Summarization"},
    {"text": "Translate the following English sentence into traditional Chinese: 'The quick brown fox jumps over the lazy dog.'", "category": "Translation"},
    {"text": "How do you say 'Where is the nearest library?' in Spanish, French, and German?", "category": "Translation"}
]

# 每類 6 筆，總計 30 筆
TEST_DATA = []
for cat in ["Chat", "Code", "QA", "Summarization", "Translation"]:
    cat_items = [d for d in REAL_DATA if d["category"] == cat]
    TEST_DATA.extend(random.choices(cat_items, k=6))

random.seed(42)
random.shuffle(TEST_DATA)

if "df_logs" not in st.session_state: st.session_state.df_logs = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ 實驗參數設定")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.subheader("💡 演示重點")
    st.info("""
    1. **消融實驗 (Ablation)**: 對比 Baseline 與 Rule-based。
    2. **持續學習 (Online Learning)**: 無需預訓練，隨實驗進行自動最佳化。
    3. **策略可視化**: 觀察 AI 如何針對不同類型文本切換 Arm。
    """)
    
    if st.button("🚀 開始實驗演示", use_container_width=True, type="primary"):
        if not api_key:
            st.error("請輸入 API Key！")
        else:
            all_logs = []
            env = RealLLMEnvironment(api_key=api_key)
            modes = ["Baseline", "Rule_Based", "LinUCB"] 
            
            with st.spinner("🤖 正在執行線上學習實驗..."):
                for mode in modes:
                    agent = LinUCB(alpha=1.0) if mode == "LinUCB" else None
                    all_logs.extend(run_experiment(TEST_DATA, mode, agent, env))
                st.session_state.df_logs = pd.DataFrame(all_logs)
            st.success("實驗完成！")
            
    if not st.session_state.df_logs.empty:
        csv = st.session_state.df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下載研究數據 (CSV)", csv, "research_results.csv", "text/csv", use_container_width=True)

if not st.session_state.df_logs.empty:
    df = st.session_state.df_logs
    
    tab1, tab2, tab3 = st.tabs(["📊 效能分析 (Metrics)", "🧠 決策熱圖 (Decision)", "📄 完整日誌 (Logs)"])
    
    with tab1:
        st.markdown("#### 1. 正式實驗數據表")
        exp_table = df.groupby("mode").agg(
            Reward=("reward", "mean"),
            Saving=("saving_ratio", lambda x: f"{x.mean()*100:.1f}%"),
            Success=("valid", lambda x: f"{x.mean()*100:.1f}%")
        ).reset_index()
        st.table(exp_table)
        
        st.markdown("#### 2. 平均獎勵收斂圖 (Avg Reward Convergence)")
        reward_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('step', title='Step'), 
            y=alt.Y('avg_reward', title='Average Reward'), 
            color='mode'
        ).properties(height=350).interactive()
        st.altair_chart(reward_chart, use_container_width=True)
        st.caption("線上學習特性：LinUCB 的平均獎勵應隨著步數增加而逐漸回升並趨於穩定。")

    with tab2:
        st.markdown("#### 3. 自適應策略分佈 (Strategy Distribution)")
        pivot_df = df[df["mode"]=="LinUCB"].groupby(["category", "arm"]).size().reset_index(name='counts')
        pivot_df['Selection Rate (%)'] = pivot_df.groupby('category')['counts'].transform(lambda x: (x / x.sum() * 100))
        
        selection_chart = alt.Chart(pivot_df).mark_bar().encode(
            x=alt.X('category:N', title='Text Category'),
            y=alt.Y('Selection Rate (%):Q', title='Frequency (%)'),
            color=alt.Color('arm:N', title='Selected Arm'),
            tooltip=['category', 'arm', 'Selection Rate (%)']
        ).properties(height=350)
        st.altair_chart(selection_chart, use_container_width=True)
        st.write("**分析重點**：觀察 AI 是否學會對 Code 類別保持低壓縮（Arm 0/1），而對 Chat 類別進行高強度壓縮（Arm 2）。")

    with tab3:
        st.dataframe(df[['step', 'mode', 'category', 'arm', 'reward', 'valid', 'llm_response']], use_container_width=True)
