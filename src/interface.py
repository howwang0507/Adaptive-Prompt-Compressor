import os
from .agent import LinUCB
from .environment import RealLLMEnvironment, SimulatedEnvironment

class LinUCBCompressor:
    """
    High-level Developer API for the Adaptive Prompt Compressor.
    Simple 3-line integration for any Python application.
    """
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", alpha=1.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.env = RealLLMEnvironment(self.api_key, model_name=model_name)
        else:
            self.env = SimulatedEnvironment()
            
        self.agent = LinUCB(n_arms=3, n_features=5, alpha=alpha)
        self.strategies = ["Conservative", "Moderate", "Aggressive"]

    def compress(self, prompt: str):
        """
        Compresses a prompt using the learned LinUCB policy.
        Returns: (compressed_prompt, strategy_name, metadata)
        """
        features = self.env.extract_features(prompt)
        arm = self.agent.select_arm(features)
        
        # In a real integration, you might want to call execute_request 
        # but here we just return the compressed version for the dev to use.
        compressed_text = self.env.compress_prompt(prompt, arm)
        
        return compressed_text, self.strategies[arm], {"arm": arm, "features": features}

    def update_policy(self, arm, features, reward):
        """
        Optional: Update the agent with real-world feedback.
        """
        self.agent.update(arm, features, reward)
