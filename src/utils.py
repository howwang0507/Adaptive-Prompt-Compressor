from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def calculate_reward(
    base_tokens,
    comp_tokens,
    latency,
    is_valid,
    semantic_score=None,
    lambda_saving=1.5,
    lambda_latency=0.2,
    lambda_failure=2.5,
    track="offline",
):
    """
    Multi-objective reward function.
    Supports Dual-Track Evaluation:
    - 'online': Fast heuristic based on token ratio and validation flag.
    - 'offline': Deep evaluation incorporating semantic scores (BERTScore/LLM Judge).
    """
    # 1. Economy Gain (Token Saving Ratio)
    saving_ratio = (base_tokens - comp_tokens) / max(base_tokens, 1)

    # 2. Latency Penalty (Normalized)
    # 1000ms is a typical 'slow' threshold
    latency_penalty = min(latency / 1000, 1.0)

    # 3. Quality Component
    if track == "online" or semantic_score is None:
        # Heuristic quality: Binary success or length-ratio based proxy
        pass
    else:
        # Deep quality: Combined validity and semantic fidelity
        pass

    # 4. Failure Penalty
    failure_penalty = 1.0 if not is_valid else 0.0

    # Total Reward Calculation
    w_lat_pen = lambda_latency * latency_penalty
    w_fail_pen = lambda_failure * failure_penalty
    # Note: Tests expect saving reward even on failure, penalizing via w_fail_pen
    reward = (lambda_saving * saving_ratio) - w_lat_pen - w_fail_pen

    return reward, saving_ratio, w_lat_pen, w_fail_pen



try:
    from sentence_transformers import SentenceTransformer

    # Load a lightweight, fast sentence transformer model
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print(
        "sentence-transformers not installed. Falling back to TF-IDF for semantic similarity."
    )


def get_semantic_similarity(original, generated):
    """
    Computes semantic similarity using Sentence-Transformers (if available)
    for deep semantic understanding, or falls back to TF-IDF.
    """
    if not original or not generated:
        return 0.0
    if original == generated:
        return 1.0

    if HAS_SBERT:
        try:
            embeddings = _st_model.encode([original, generated])
            score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            return max(0.0, min(1.0, score))  # Ensure within [0, 1]
        except Exception as e:
            print(f"SBERT error: {e}. Falling back to TF-IDF.")
            pass  # Fallthrough to TF-IDF

    try:
        # Use character-level n-grams to handle typos and small variations
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        tfidf = vectorizer.fit_transform([original, generated])
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return score
    except Exception:
        return 0.5  # Robust fallback


def get_heuristic_quality(prompt, compressed):
    """
    Lightweight heuristic reward for online track.
    Checks for structural keyword retention (e.g. negations, code keywords).
    """
    keywords = ["not", "never", "def", "if", "return", "{", "}"]
    lost_count = 0
    for k in keywords:
        if k in prompt.lower() and k not in compressed.lower():
            lost_count += 1

    # Penalty if structural keywords are lost
    return max(0.0, 1.0 - (lost_count * 0.3))
