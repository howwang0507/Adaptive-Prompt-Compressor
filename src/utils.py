import random
import numpy as np

# Note: In a real production environment, you would use:
# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_reward(base_tokens, comp_tokens, latency, valid, semantic_score=1.0):
    """
    Enhanced Multi-Objective Reward Function.
    Includes semantic_score (0.0 to 1.0) derived from BERTScore/Cosine Similarity.
    """
    saving = (base_tokens - comp_tokens) / max(base_tokens, 1)
    latency_pen = 0.2 * (latency / 1000.0) 
    
    # Failure Penalty is now weighted by semantic validity
    # If invalid (valid=False), penalty is maximum. 
    # If valid, reward is scaled by how semantically close it is to the original response.
    fail_pen = 2.5 if not valid else (2.5 * (1.0 - semantic_score))
    
    reward = (1.5 * saving - latency_pen - fail_pen)
    return reward, saving, latency_pen, fail_pen

def get_test_dataset(real_data, num_samples=50):
    dataset = []
    categories = list(set(d["category"] for d in real_data))
    for cat in categories:
        cat_items = [d for d in real_data if d["category"] == cat]
        dataset.extend(random.choices(cat_items, k=num_samples // len(categories)))
    random.seed(42)
    random.shuffle(dataset)
    return dataset

def get_semantic_similarity(text1, text2):
    """
    Placeholder for BERTScore/Sentence-Transformer similarity.
    In a real run, this would compare the LLM response of the raw prompt 
    vs the compressed prompt.
    """
    # For simulation purposes, we return a high similarity for small arms, 
    # and lower/random similarity for aggressive arms.
    return np.random.uniform(0.85, 1.0)
