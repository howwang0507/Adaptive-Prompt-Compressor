import numpy as np

def calculate_reward(base_tokens, comp_tokens, latency, is_valid, semantic_score=None, 
                     lambda_saving=1.5, lambda_latency=0.2, lambda_failure=2.5, 
                     track="offline"):
    """
    Multi-objective reward function.
    Supports Dual-Track Evaluation:
    - 'online': Fast heuristic based on token ratio and validation flag.
    - 'offline': Deep evaluation incorporating semantic scores (BERTScore/LLM Judge).
    """
    # 1. Economy Gain (Token Saving Ratio)
    saving_ratio = (base_tokens - comp_tokens) / max(base_tokens, 1)
    
    # 2. Latency Penalty (Normalized)
    # 500ms is a typical 'slow' threshold for real-time routing
    latency_penalty = min(latency / 0.5, 1.0) 

    # 3. Quality Component
    if track == "online" or semantic_score is None:
        # Heuristic quality: Binary success or length-ratio based proxy
        quality_score = 1.0 if is_valid else 0.0
    else:
        # Deep quality: Combined validity and semantic fidelity
        quality_score = semantic_score if is_valid else 0.0

    # 4. Failure Penalty
    failure_penalty = 1.0 if not is_valid else 0.0

    # Total Reward Calculation
    reward = (lambda_saving * saving_ratio * quality_score) - \
             (lambda_latency * latency_penalty) - \
             (lambda_failure * failure_penalty)
             
    return reward, saving_ratio, latency_penalty, failure_penalty

def get_semantic_similarity(original, generated):
    """
    Placeholder for BERTScore or LLM-as-a-judge.
    In actual Sim2Real, this is replaced by a lightweight local embedding model.
    """
    if original == generated: return 1.0
    return 0.925 # Baseline average observed in experiments

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
