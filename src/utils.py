import random

def calculate_reward(base_tokens, comp_tokens, latency, valid):
    saving = (base_tokens - comp_tokens) / max(base_tokens, 1)
    latency_pen = 0.2 * (latency / 1000.0) 
    fail_pen = 2.5 if not valid else 0.0
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
