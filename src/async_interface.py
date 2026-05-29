import asyncio
import os
from typing import List, Tuple, Dict, Any
from .agent import LinUCB
from .environment import RealLLMEnvironment, SimulatedEnvironment


class AsyncLinUCBCompressor:
    """
    Asynchronous, high-throughput developer API for the Adaptive Prompt Compressor.
    Supports batch processing and auto-fallback for production-grade reliability.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-1.5-flash",
        alpha: float = 1.0,
        fallback_threshold: float = 0.6,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.env = RealLLMEnvironment(self.api_key, model_name=model_name)
        else:
            self.env = SimulatedEnvironment()

        self.agent = LinUCB(n_arms=3, n_features=5, alpha=alpha)
        self.fallback_threshold = fallback_threshold
        self.strategies = ["Conservative", "Moderate", "Aggressive"]

    async def compress_async(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Asynchronously selects a strategy and returns the compressed prompt.
        """
        # Feature extraction (could be heavy, run in executor if needed)
        features = await asyncio.to_thread(self.env.extract_features, prompt)
        arm = self.agent.select_arm(features)

        # Simulated/Actual compression logic
        compressed_text = await asyncio.to_thread(self.env.compress_prompt, prompt, arm)

        return compressed_text, self.strategies[arm], {"arm": arm, "features": features}

    async def execute_with_fallback(self, prompt: str) -> Dict[str, Any]:
        """
        Executes a prompt with compression and automatically falls back to the
        original prompt if semantic quality or API failure occurs.
        """
        compressed_text, strategy, meta = await self.compress_async(prompt)

        try:
            # Attempt with compressed prompt
            result = await asyncio.to_thread(
                self.env.execute_request, prompt, meta["arm"]
            )

            # Semantic check (simplistic threshold check)
            # In real production, this would be a more complex LLM-as-a-judge or BERTScore
            if result.get("semantic_score", 1.0) < self.fallback_threshold:
                print(
                    f"⚠️ Quality below threshold ({result['semantic_score']}). Falling back..."
                )
                result = await asyncio.to_thread(
                    self.env.execute_request, prompt, arm=0
                )  # Arm 0 is raw/conservative
                result["fallback_triggered"] = True
            else:
                result["fallback_triggered"] = False

        except Exception as e:
            print(f"❌ Execution failed: {e}. Falling back to raw prompt...")
            result = await asyncio.to_thread(self.env.execute_request, prompt, arm=0)
            result["fallback_triggered"] = True
            result["error"] = str(e)

        result["strategy_used"] = strategy
        return result

    async def compress_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Processes multiple prompts in parallel for high throughput.
        """
        tasks = [self.execute_with_fallback(p) for p in prompts]
        return await asyncio.gather(*tasks)
