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
        
        # 增加重試機制與更長的等待時間以應對免費版 API 限制 (15 RPM)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 每個 Request 至少間隔 4.5 秒，確保每分鐘不超過 13-14 次
                time.sleep(4.5) 
                response = self.model.generate_content(
                    f"Briefly respond to this: {compressed_text}",
                    generation_config={"max_output_tokens": 150}
                )
                answer = response.text
                break # 成功則跳出重試迴圈
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"[WARNING] Rate Limit Hit. Retrying in 10s... (Attempt {attempt+1})")
                    time.sleep(10) # 遇到 429 額外多等 10 秒
                    continue
                answer = f"API Error: {str(e)}"
                is_valid = False
                print(f"[ERROR] API Call Failed: {str(e)}")
                break
        
        if is_valid:
            # 強化驗證邏輯
            confusion_flags = ["i don't understand", "not sure", "unclear", "clarify", "聽不懂", "不清楚", "error", "blocked"]
            if any(flag in answer.lower() for flag in confusion_flags):
                is_valid = False
            
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in answer)
            if has_chinese:
                if len(answer.strip()) < 3: is_valid = False
            else:
                if len(answer.split()) < 2: is_valid = False
            
            if len(answer.strip()) == 0: is_valid = False
                
        clean_ans = answer[:50].replace('\n', ' ')
        print(f"[DEBUG] Arm: {arm} | Valid: {is_valid} | Response: {clean_ans}...")
            
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

# 實驗資料集 (擴充版 - 涵蓋更多場景)
REAL_DATA = [
    # --- Chat Category ---
    {"text": "Could you please explain to me in detail what the weather is usually like in Tokyo during the spring season?", "category": "Chat"},
    {"text": "Tell me about the history of the Eiffel Tower, including when it was built and who designed it.", "category": "Chat"},
    {"text": "What are some fun things to do in San Francisco for a first-time visitor?", "category": "Chat"},
    {"text": "Can you give me a list of healthy snacks that are easy to prepare at home?", "category": "Chat"},
    {"text": "I'm feeling a bit stressed today. Could you suggest some relaxation techniques?", "category": "Chat"},
    {"text": "What's the best way to learn a new language quickly and effectively?", "category": "Chat"},
    
    # --- Code Category ---
    {"text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)", "category": "Code"},
    {"text": "Write a Python function to scrape news headlines from a website using BeautifulSoup.", "category": "Code"},
    {"text": "const factorial = (n) => n === 0 ? 1 : n * factorial(n - 1);", "category": "Code"},
    {"text": "Explain how a React useEffect hook works with a simple example.", "category": "Code"},
    {"text": "SELECT name, age FROM users WHERE status = 'active' ORDER BY age DESC LIMIT 10;", "category": "Code"},
    {"text": "How do I implement a binary search algorithm in Java?", "category": "Code"},

    # --- QA Category ---
    {"text": "Who was the first person to walk on the moon, and what was the name of the mission?", "category": "QA"},
    {"text": "What are the primary causes of climate change, and how do greenhouse gases contribute?", "category": "QA"},
    {"text": "Who wrote the play 'Romeo and Juliet' and in what century was it written?", "category": "QA"},
    {"text": "What is the capital of Australia and what are some of its major landmarks?", "category": "QA"},
    {"text": "How does photosynthesis work in plants at a cellular level?", "category": "QA"},
    {"text": "What are the main differences between a virus and a bacteria?", "category": "QA"},

    # --- Summarization Category ---
    {"text": "The Industrial Revolution was a period of global economic transition towards more efficient manufacturing processes. It began in Great Britain and then spread to other parts of the world. Please summarize this passage in one sentence.", "category": "Summarization"},
    {"text": "A supernova is a powerful and luminous stellar explosion occurring during the last evolutionary stages of a star. It can briefly outshine an entire galaxy. Provide a brief summary.", "category": "Summarization"},
    {"text": "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. Summarize this concept for a non-expert.", "category": "Summarization"},
    {"text": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. Summarize its historical significance.", "category": "Summarization"},

    # --- Translation Category ---
    {"text": "Translate the following English sentence into traditional Chinese: 'The quick brown fox jumps over the lazy dog.'", "category": "Translation"},
    {"text": "How do you say 'Where is the nearest library?' in Spanish, French, and German?", "category": "Translation"},
    {"text": "Translate 'Artificial Intelligence is changing the world' into Japanese and Korean.", "category": "Translation"},
    {"text": "What is the Italian translation for 'I would like to order a pizza with extra cheese'?", "category": "Translation"}
]

# 每類 10 筆，總計 50 筆 (根據現有擴充)
def get_test_dataset(num_samples=50):
    dataset = []
    for cat in ["Chat", "Code", "QA", "Summarization", "Translation"]:
        cat_items = [d for d in REAL_DATA if d["category"] == cat]
        dataset.extend(random.choices(cat_items, k=num_samples // 5))
    random.seed(42)
    random.shuffle(dataset)
    return dataset

if "df_logs" not in st.session_state: st.session_state.df_logs = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ 實驗參數設定")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.subheader("🛠️ 數據源設定")
    data_source = st.radio("選擇數據來源", ["內建擴充資料集 (50 筆)", "自定義輸入 Prompt"])
    
    custom_prompt = ""
    if data_source == "自定義輸入 Prompt":
        custom_prompt = st.text_area("輸入自定義文本 (多行輸入)", height=150, placeholder="在此輸入您想測試的 Prompt...")
        st.caption("注意：自定義模式會針對該 Prompt 重複執行以觀察學習曲線。")
    
    st.markdown("---")
    st.subheader("💡 演示重點")
    st.info("""
    1. **消融實驗 (Ablation)**: 對比 Baseline 與 Rule-based。
    2. **持續學習 (Online Learning)**: 無需預訓練，隨實驗進行自動最佳化。
    3. **策略可視化**: 觀察 AI 如何針對不同類型文本切換 Arm。
    """)
    
    st.markdown("---")
    st.subheader("🧪 實驗模式選擇")
    selected_modes = st.multiselect("選擇要執行的模式", ["Baseline", "Rule_Based", "LinUCB"], default=["LinUCB"])
    st.caption("提示：配額有限時，建議一次只勾選一個模式執行。")

    if st.button("🚀 開始實驗演示", use_container_width=True, type="primary"):
        if not api_key:
            st.error("請輸入 API Key！")
        elif not selected_modes:
            st.warning("請至少選擇一個模式！")
        else:
            all_logs = []
            env = RealLLMEnvironment(api_key=api_key)
            
            # 準備數據
            if data_source == "自定義輸入 Prompt" and custom_prompt:
                current_test_data = [{"text": custom_prompt, "category": "Custom"}] * 20
            else:
                current_test_data = get_test_dataset(50)
            
            with st.spinner(f"🤖 正在執行 {len(current_test_data) * len(selected_modes)} 步實驗..."):
                for mode in selected_modes:
                    agent = LinUCB(alpha=1.0) if mode == "LinUCB" else None
                    all_logs.extend(run_experiment(current_test_data, mode, agent, env))
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
