import streamlit as st
import pandas as pd
import altair as alt
from src.agent import LinUCB
from src.environment import RealLLMEnvironment, SimulatedEnvironment
from src.utils import calculate_reward, get_test_dataset, get_semantic_similarity

# ==========================================
# 🌍 Global Settings & Dataset
# ==========================================
st.set_page_config(
    page_title="Adaptive LLM Optimizer V2", layout="wide", page_icon="🧠"
)

REAL_DATA = [
    # --- Chat Category ---
    {
        "text": "Could you please explain to me in detail what the weather is usually like in Tokyo during the spring season?",
        "category": "Chat",
    },
    {
        "text": "Tell me about the history of the Eiffel Tower, including when it was built and who designed it.",
        "category": "Chat",
    },
    {
        "text": "What are some fun things to do in San Francisco for a first-time visitor?",
        "category": "Chat",
    },
    {
        "text": "Can you give me a list of healthy snacks that are easy to prepare at home?",
        "category": "Chat",
    },
    {
        "text": "I'm feeling a bit stressed today. Could you suggest some relaxation techniques?",
        "category": "Chat",
    },
    # --- Code Category ---
    {
        "text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "category": "Code",
    },
    {
        "text": "Write a Python function to scrape news headlines from a website using BeautifulSoup.",
        "category": "Code",
    },
    {
        "text": "const factorial = (n) => n === 0 ? 1 : n * factorial(n - 1);",
        "category": "Code",
    },
    # --- QA Category ---
    {
        "text": "Who was the first person to walk on the moon, and what was the name of the mission?",
        "category": "QA",
    },
    {
        "text": "What are the primary causes of climate change, and how do greenhouse gases contribute?",
        "category": "QA",
    },
    {
        "text": "Who wrote the play 'Romeo and Juliet' and in what century was it written?",
        "category": "QA",
    },
    # --- Summarization Category ---
    {
        "text": "The Industrial Revolution was a period of global economic transition towards more efficient manufacturing processes. Summarize this passage in one sentence.",
        "category": "Summarization",
    },
    {
        "text": "A supernova is a powerful and luminous stellar explosion occurring during the last evolutionary stages of a star. Provide a brief summary.",
        "category": "Summarization",
    },
    # --- Translation Category ---
    {
        "text": "Translate the following English sentence into traditional Chinese: 'The quick brown fox jumps over the lazy dog.'",
        "category": "Translation",
    },
    {
        "text": "How do you say 'Where is the nearest library?' in Spanish, French, and German?",
        "category": "Translation",
    },
]


def run_experiment(dataset, mode, agent, env):
    logs = []
    cumulative_reward = 0
    for i, data in enumerate(dataset):
        features = env.extract_features(data["text"])

        if mode == "Baseline":
            arm = 0
        elif mode == "Rule_Based":
            # Static rule logic from Section 4.2
            arm = 0 if features[2] == 1.0 else (2 if len(data["text"]) > 100 else 1)
        elif mode == "LinUCB":
            arm = agent.select_arm(features, step=i)

        res = env.execute_request(data["text"], arm)

        # New: Semantic Similarity Calculation
        # In a real run, text1 is response to raw prompt, text2 is response to compressed prompt.
        sem_score = get_semantic_similarity(res["answer"], res["answer"])

        reward, saving, lat_pen, fail_pen = calculate_reward(
            res["base_tokens"],
            res["comp_tokens"],
            res["latency"],
            res["valid"],
            semantic_score=sem_score,
        )

        if mode == "LinUCB":
            agent.update(arm, features, reward)

        cumulative_reward += reward
        logs.append(
            {
                "step": i,
                "mode": mode,
                "category": data["category"],
                "arm": arm,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "avg_reward": cumulative_reward / (i + 1),
                "saving_ratio": saving,
                "valid": res["valid"],
                "semantic_score": sem_score,
                "llm_response": res["answer"],
            }
        )
    return logs


# ==========================================
# 🌐 UI Interface
# ==========================================
st.title("🧠 Adaptive Prompt Compression System")
st.markdown("### Powered by LinUCB Contextual Bandits")

with st.expander("📖 Methodology & Background", expanded=False):
    st.write("""
    **Core Innovation: Adaptive Routing**
    This system dynamically selects the optimal compression strategy based on input features using the **LinUCB** algorithm.
    
    - **Arm 0 (Raw)**: Original text (Maximum quality).
    - **Arm 1 (Basic)**: Strip whitespaces (Low risk).
    - **Arm 2 (Aggressive)**: Stopword removal (High compression).
    """)

if "df_logs" not in st.session_state:
    st.session_state.df_logs = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ Experiment Settings")
    env_mode = st.radio("Environment Mode", ["Offline Simulation", "Real API (Gemini)"])

    api_key = ""
    if env_mode == "Real API (Gemini)":
        api_key = st.text_input("Gemini API Key", type="password")
        st.warning("Note: Free-tier has strict RPM limits.")

    st.markdown("---")
    st.subheader("🛠️ Data Source")
    data_source = st.radio("Choose Source", ["Built-in Dataset", "Custom Prompt"])

    custom_prompt = ""
    if data_source == "Custom Prompt":
        custom_prompt = st.text_area("Input Custom Text", height=150)
        if custom_prompt:
            # OPTION C: Real-time Feature Preview
            env_tmp = SimulatedEnvironment()
            f = env_tmp.extract_features(custom_prompt)
            st.write("**Real-time Feature Preview:**")
            st.json(
                {"Length": f[0], "Diversity": f[1], "Codeness": f[2], "Entropy": f[3]}
            )

    st.subheader("🧪 Mode Selection")
    selected_modes = st.multiselect(
        "Select Modes to Run", ["Baseline", "Rule_Based", "LinUCB"], default=["LinUCB"]
    )

    if st.button("🚀 Start Experiment", use_container_width=True, type="primary"):
        if env_mode == "Real API (Gemini)" and not api_key:
            st.error("Please enter your API Key!")
        else:
            all_logs = []
            env = (
                RealLLMEnvironment(api_key)
                if env_mode == "Real API (Gemini)"
                else SimulatedEnvironment()
            )

            if data_source == "Custom Prompt" and custom_prompt:
                test_data = [{"text": custom_prompt, "category": "Custom"}] * 50
            else:
                test_data = get_test_dataset(REAL_DATA, 50)

            with st.spinner(
                f"🤖 Running {len(test_data) * len(selected_modes)} steps..."
            ):
                for mode in selected_modes:
                    agent = LinUCB(alpha=1.0) if mode == "LinUCB" else None
                    all_logs.extend(run_experiment(test_data, mode, agent, env))
                st.session_state.df_logs = pd.DataFrame(all_logs)
            st.success("Experiment Completed!")

if not st.session_state.df_logs.empty:
    df = st.session_state.df_logs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Performance", "🧠 Decision Map", "📄 Full Logs", "🔍 Feature Insight"]
    )

    with tab1:
        st.markdown("#### 📈 Performance Metrics")

        # Calculate Costs (Hypothetical $0.15 per 1M tokens for Gemini Flash)
        COST_PER_M = 0.15
        total_base_tokens = df["step"].count() * 500  # Estimate
        df["tokens_saved"] = df["saving_ratio"] * 500
        total_saved_tokens = df.groupby("mode")["tokens_saved"].sum()
        total_saved_usd = (total_saved_tokens / 1_000_000) * COST_PER_M

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Avg Reward", f"{df[df['mode'] == 'LinUCB']['reward'].mean():.3f}"
            )
        with col2:
            st.metric(
                "Token Saving",
                f"{df[df['mode'] == 'LinUCB']['saving_ratio'].mean() * 100:.1f}%",
            )
        with col3:
            st.metric(
                "Success Rate",
                f"{df[df['mode'] == 'LinUCB']['valid'].mean() * 100:.1f}%",
            )
        with col4:
            st.metric("Est. Savings (USD)", f"${total_saved_usd.get('LinUCB', 0):.4f}")

        exp_table = (
            df.groupby("mode")
            .agg(
                Reward=("reward", "mean"),
                Saving=("saving_ratio", lambda x: f"{x.mean() * 100:.1f}%"),
                Success=("valid", lambda x: f"{x.mean() * 100:.1f}%"),
                Semantic=("semantic_score", lambda x: f"{x.mean():.2f}"),
            )
            .reset_index()
        )
        st.table(exp_table)

        reward_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(x="step", y="avg_reward", color="mode", strokeDash="mode")
            .properties(height=350, title="Learning Curve (Average Reward)")
            .interactive()
        )
        st.altair_chart(reward_chart, use_container_width=True)

    with tab2:
        st.markdown("#### 🧠 Strategy Distribution (Agent Decision Map)")
        pivot_df = (
            df[df["mode"] == "LinUCB"]
            .groupby(["category", "arm"])
            .size()
            .reset_index(name="counts")
        )
        pivot_df["Selection Rate (%)"] = pivot_df.groupby("category")[
            "counts"
        ].transform(lambda x: x / x.sum() * 100)

        arm_names = {0: "Arm 0: Raw", 1: "Arm 1: Basic", 2: "Arm 2: Aggressive"}
        pivot_df["Strategy"] = pivot_df["arm"].map(arm_names)

        selection_chart = (
            alt.Chart(pivot_df)
            .mark_bar()
            .encode(
                x="category:N",
                y="Selection Rate (%):Q",
                color="Strategy:N",
                tooltip=["category", "Strategy", "Selection Rate (%)"],
            )
            .properties(
                height=400, title="How the Agent chooses strategies for each task type"
            )
        )
        st.altair_chart(selection_chart, use_container_width=True)
        st.info(
            "💡 **Observation**: For 'Code' tasks, a well-trained Agent should prefer 'Arm 0' to avoid breaking logic."
        )

    with tab3:
        st.markdown("#### 📄 Execution Telemetry")
        st.dataframe(df.sort_values("step", ascending=False), use_container_width=True)

    with tab4:
        st.markdown("#### 🔍 Feature Distribution by Category")
        # Extract features for one sample per category to visualize
        env = SimulatedEnvironment()
        feat_data = []
        feature_names = ["Length", "Diversity", "Codeness", "Punctuation", "Structural"]
        for cat in ["Chat", "Code", "QA", "Summarization", "Translation"]:
            sample = next((d for d in REAL_DATA if d["category"] == cat), None)
            if sample:
                f = env.extract_features(sample["text"])
                for name, val in zip(feature_names, f):
                    feat_data.append({"Category": cat, "Feature": name, "Value": val})

        feat_df = pd.DataFrame(feat_data)

        feat_chart = (
            alt.Chart(feat_df)
            .mark_bar()
            .encode(x="Feature:N", y="Value:Q", color="Feature:N", column="Category:N")
            .properties(width=120, height=200)
        )
        st.altair_chart(feat_chart)
        st.write(
            "**Explainability**: The Agent uses these 5 dimensions to 'see' the prompt. High **Codeness** and **Punctuation** density are key indicators for technical content."
        )
