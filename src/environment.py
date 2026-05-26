import time
import re
import os
import numpy as np
import google.generativeai as genai

# Try to import Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False

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
        if not text or len(text.strip()) == 0:
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        words = text.split()
        num_words = len(words)
        s_1 = min(len(text) / 1000.0, 1.0)
        s_2 = len(set(words)) / num_words if num_words > 0 else 0.0
        code_pattern = re.compile(r'\b(def|function|class|return|import|const|let)\b|[{}:;=]')
        s_3 = 1.0 if code_pattern.search(text) else 0.0
        avg_word_length = sum(len(w) for w in words) / num_words if num_words > 0 else 0.0
        s_4 = min(avg_word_length / 20.0, 1.0)
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
        answer, is_valid = "", True
        for attempt in range(3):
            try:
                time.sleep(5.0) 
                response = self.model.generate_content(f"Briefly respond to this: {compressed_text}", generation_config={"max_output_tokens": 150})
                answer = response.text
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg and attempt < 2:
                    wait_time = 60
                    match = re.search(r"retry in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    time.sleep(wait_time); continue
                answer, is_valid = f"API Error: {err_msg}", False; break
        
        if is_valid:
            confusion_flags = ["i don't understand", "not sure", "unclear", "听不懂", "不清楚", "error", "blocked"]
            if any(flag in answer.lower() for flag in confusion_flags): is_valid = False
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in answer)
            if has_chinese:
                if len(answer.strip()) < 3: is_valid = False
            else:
                if len(answer.split()) < 2: is_valid = False
            if len(answer.strip()) == 0: is_valid = False
        return {"base_tokens": base_tokens, "comp_tokens": comp_tokens, "latency": (time.time() - start_time) * 1000, "valid": is_valid, "answer": answer}

class VertexAIEnvironment(RealLLMEnvironment):
    def __init__(self, project_id, key_path, location="us-central1"):
        if not HAS_VERTEX:
            raise ImportError("google-cloud-aiplatform not installed.")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        
        # System instruction helps the model handle malformed/compressed inputs
        self.model = VertexGenerativeModel(
            "gemini-1.5-flash-001",
            system_instruction="You are a research assistant. Respond concisely to the user's prompt, even if it is highly compressed or missing common stopwords."
        )
        self.stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "of", "to", "in", "for", "with", "on", "at", "by", "this", "that"}

    def execute_request(self, text, arm):
        compressed_text = self.compress_prompt(text, arm)
        start_time = time.time()
        answer, is_valid = "", True
        
        # Max unblocking: BLOCK_NONE for all categories
        from vertexai.generative_models import HarmCategory, HarmBlockThreshold
        safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            # Use a clearer user-message structure
            response = self.model.generate_content(
                compressed_text,
                safety_settings=safety_config,
                generation_config={"temperature": 0.2, "max_output_tokens": 100}
            )
            answer = response.text
        except Exception as e:
            # Handle block cases explicitly
            answer, is_valid = f"Vertex Error: {str(e)}", False
            
        # Run validation logic (length check etc.)
        if is_valid:
            if len(answer.strip()) < 2: is_valid = False
            
        return {"base_tokens": len(text)//4, "comp_tokens": len(compressed_text)//4, 
                "latency": (time.time() - start_time) * 1000, "valid": is_valid, "answer": answer}

class SimulatedEnvironment:
    def __init__(self):
        self.stop_words = {"a", "an", "the"} 
    def extract_features(self, text):
        return RealLLMEnvironment.extract_features(self, text)
    def execute_request(self, text, arm):
        is_code = "def " in text or "{" in text
        success_prob = 0.95 if arm == 0 else (0.85 if arm == 1 else (0.4 if is_code else 0.7))
        valid = np.random.random() < success_prob
        base_tokens = len(text) // 4
        comp_tokens = int(base_tokens * (1.0 if arm == 0 else (0.9 if arm == 1 else 0.6)))
        return {"base_tokens": base_tokens, "comp_tokens": comp_tokens, "latency": np.random.uniform(500, 1500), "valid": valid, "answer": "SIMULATED RESPONSE"}
