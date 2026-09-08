import os
from .agent import LinUCB
from .environment import (
    AnthropicEnvironment,
    OpenAIEnvironment,
    RealLLMEnvironment,
    SimulatedEnvironment,
)


class LinUCBCompressor:
    """
    High-level Developer API for the Adaptive Prompt Compressor.
    Simple 3-line integration for any Python application with native OpenAI support.
    """

    def __init__(self, api_key=None, provider="auto", model_name=None, alpha=1.0):
        self.provider = provider
        openai_key = api_key if provider == "openai" else (os.environ.get("OPENAI_API_KEY") or api_key)
        gemini_key = api_key if provider == "gemini" else (os.environ.get("GEMINI_API_KEY") or api_key)
        anthropic_key = api_key if provider == "anthropic" else (os.environ.get("ANTHROPIC_API_KEY") or api_key)

        if (provider == "openai" or (provider == "auto" and os.environ.get("OPENAI_API_KEY"))) and openai_key:
            self.provider = "openai"
            self.api_key = openai_key
            self.env = OpenAIEnvironment(self.api_key, model_name=model_name or "gpt-4o-mini")
        elif (provider == "gemini" or (provider == "auto" and os.environ.get("GEMINI_API_KEY"))) and gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            self.env = RealLLMEnvironment(self.api_key, model_name=model_name or "gemini-1.5-flash")
        elif (provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY"))) and anthropic_key:
            self.provider = "anthropic"
            self.api_key = anthropic_key
            self.env = AnthropicEnvironment(self.api_key, model_name=model_name or "claude-3-haiku-20240307")
        else:
            self.provider = "simulation" if provider == "auto" else f"{provider} (offline simulation)"
            self.api_key = None
            self.env = SimulatedEnvironment()

        n_features = len(self.env.extract_features("init"))
        self.agent = LinUCB(n_arms=3, n_features=n_features, alpha=alpha)
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
