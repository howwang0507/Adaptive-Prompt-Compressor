import time
import re
import numpy as np
import google.generativeai as genai

class RealLLMEnvironment:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model_name = 'gemini-1.5-flash'
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_models = ['gemini-1.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash']
            selected = None
            for target in target_models:
                if any(target in m for m in available_models):
                    selected = next(m for m in available_models if target in m)
                    break
            if selected: self.model_name = selected
        except: pass
        self.model = genai.GenerativeModel(self.model_name)
        self.stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "of", "to", "in", "for", "with", "on", "at", "by", "this", "that"}

    def extract_features(self, text):
        """
        Refined Feature Extraction based on Research Paper Section 3.1.
        Returns a 5-dimensional vector [length, diversity, codeness, entropy, bias].
        """
        if not text or len(text.strip()) == 0:
            return [0.0, 0.0, 0.0, 0.0, 1.0]

        words = text.split()
        num_words = len(words)

        # 1. Normalized length (clamped at 1000)
        s_1 = min(len(text) / 1000.0, 1.0)

        # 2. Lexical diversity (Unique/Total)
        s_2 = len(set(words)) / num_words if num_words > 0 else 0.0

        # 3. Structural flag (Codeness) - Refined Regex
        code_pattern = re.compile(r'\b(def|function|class|return|import|const|let)\b|[{}:;=]')
        s_3 = 1.0 if code_pattern.search(text) else 0.0

        # 4. Semantic entropy approximation (Avg word length)
        avg_word_length = sum(len(w) for w in words) / num_words if num_words > 0 else 0.0
        s_4 = min(avg_word_length / 20.0, 1.0)

        # 5. Bias term (Constant 1.0 for LinUCB intercept)
        return [s_1, s_2, s_3, s_4, 1.0]

    def compress_prompt(self, text, arm):
        if arm == 0: return text 
        if arm == 1: return re.sub(r'\s+', ' ', text).strip()
        if arm == 2:
            words = text.split()
            kept_words = [w for w in words if w.lower() not in self.stop_words]
            return " ".join(kept_words)

    def execute_request(self, text, arm):
        try: base_tokens = self.model.count_tokens(text).total_tokens
        except: base_tokens = len(text) // 4
        
        compressed_text = self.compress_prompt(text, arm)
        try: comp_tokens = self.model.count_tokens(compressed_text).total_tokens
        except: comp_tokens = len(compressed_text) // 4
        
        start_time = time.time()
        answer = ""
        is_valid = True
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(5.0) 
                response = self.model.generate_content(f"Briefly respond to this: {compressed_text}", generation_config={"max_output_tokens": 150})
                answer = response.text
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg and attempt < max_retries - 1:
                    wait_time = 60
                    match = re.search(r"retry in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    time.sleep(wait_time)
                    continue
                answer = f"API Error: {err_msg}"
                is_valid = False
                break
        
        if is_valid:
            confusion_flags = ["i don't understand", "not sure", "unclear", "听不懂", "不清楚", "error", "blocked"]
            if any(flag in answer.lower() for flag in confusion_flags): is_valid = False
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in answer)
            if has_chinese:
                if len(answer.strip()) < 3: is_valid = False
            else:
                if len(answer.split()) < 2: is_valid = False
            if len(answer.strip()) == 0: is_valid = False
                
        return {"base_tokens": base_tokens, "comp_tokens": comp_tokens, 
                "latency": (time.time() - start_time) * 1000, "valid": is_valid, "answer": answer}

class SimulatedEnvironment:
    def __init__(self):
        self.stop_words = {"a", "an", "the"} # Simplified

    def extract_features(self, text):
        return RealLLMEnvironment.extract_features(self, text)

    def execute_request(self, text, arm):
        # Simulate success/failure based on arm and content
        # Arm 2 is riskier, especially if it looks like code
        is_code = "def " in text or "{" in text
        success_prob = 0.95 if arm == 0 else (0.85 if arm == 1 else (0.4 if is_code else 0.7))
        valid = np.random.random() < success_prob
        
        base_tokens = len(text) // 4
        comp_tokens = int(base_tokens * (1.0 if arm == 0 else (0.9 if arm == 1 else 0.6)))
        latency = np.random.uniform(500, 1500)
        
        return {"base_tokens": base_tokens, "comp_tokens": comp_tokens, 
                "latency": latency, "valid": valid, "answer": "SIMULATED RESPONSE"}
