import numpy as np
import threading
import os
import json
import logging
from typing import List, Tuple, Optional, Dict

try:
    import redis

    REDIS_URL = os.environ.get("REDIS_URL")
    redis_client = redis.Redis.from_url(REDIS_URL) if REDIS_URL else None
except ImportError:
    redis_client = None


class OnlineScaler:
    """
    Implements Welford's algorithm for online mean and variance estimation.
    """

    def __init__(self, n_features: int):
        self.n: int = 0
        self.mean: np.ndarray = np.zeros(n_features)
        self.M2: np.ndarray = np.zeros(n_features)
        self._lock: threading.Lock = threading.Lock()

    def update(self, x: np.ndarray) -> None:
        with self._lock:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        return self.M2 / (self.n - 1) if self.n > 1 else np.ones_like(self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x
        std = np.sqrt(self.var)
        std[std == 0] = 1.0
        return (x - self.mean) / std

    def is_ood(self, x: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Out-of-Distribution (OOD) check:
        Returns True if any feature is more than 'threshold' standard deviations away.
        """
        if self.n < 10:
            return False  # Need warm-up
        z_scores = np.abs((x - self.mean) / (np.sqrt(self.var) + 1e-6))
        return bool(np.any(z_scores > threshold))


class LinUCB:
    """
    Contextual Multi-Armed Bandit using the LinUCB algorithm.
    Enhanced for Numerical Stability, Thread-Safety, OOD Protection, and Redis Distributed Sync.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        gamma: float = 0.99,
        lambda_reg: float = 1.0,
        domain_priors: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.n_arms: int = n_arms
        self.n_features: int = n_features
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.use_redis = redis_client is not None

        # Ridge Regularization or Meta-Learning Warm Start
        if domain_priors:
            # Zero-Shot Domain Adaptation: Load pre-trained covariance and reward matrices
            self.A_inv: List[np.ndarray] = [domain_priors[a][0] for a in range(n_arms)]
            self.b: List[np.ndarray] = [domain_priors[a][1] for a in range(n_arms)]
            print("🚀 Meta-Learning: Initialized with Zero-Shot Domain Priors.")
        else:
            self.A_inv: List[np.ndarray] = [
                np.identity(n_features) / lambda_reg for _ in range(n_arms)
            ]
            self.b: List[np.ndarray] = [
                np.zeros((n_features, 1)) for _ in range(n_arms)
            ]

        if self.use_redis:
            self._sync_from_redis()

        # Scaler for online normalization

        self.scaler: OnlineScaler = OnlineScaler(n_features)

        # Thread safety lock
        self._lock: threading.Lock = threading.Lock()

    def _sync_from_redis(self):
        """Fetches latest matrices from Redis parameter server."""
        if not self.use_redis:
            return
        try:
            for a in range(self.n_arms):
                a_data = redis_client.get(f"linucb:A_inv:{a}")
                b_data = redis_client.get(f"linucb:b:{a}")
                if a_data:
                    self.A_inv[a] = np.array(json.loads(a_data))
                if b_data:
                    self.b[a] = np.array(json.loads(b_data))
        except Exception as e:
            logging.error(f"Redis sync failed: {e}")

    def _sync_to_redis(self, arm: int):
        """Pushes latest matrices to Redis parameter server."""
        if not self.use_redis:
            return
        try:
            redis_client.set(
                f"linucb:A_inv:{arm}", json.dumps(self.A_inv[arm].tolist())
            )
            redis_client.set(f"linucb:b:{arm}", json.dumps(self.b[arm].tolist()))
        except Exception as e:
            logging.error(f"Redis write failed: {e}")

    def select_arm(self, features: np.ndarray, step: Optional[int] = None) -> int:
        """
        Selects the optimal arm using UCB strategy.
        """
        if self.use_redis:
            self._sync_from_redis()

        if isinstance(features, list):
            features = np.array(features)

        if self.scaler.is_ood(features):
            # Fallback to Arm 0 (Conservative) for OOD inputs
            return 0

        x = self.scaler.transform(features).reshape(-1, 1)
        p = np.zeros(self.n_arms)

        with self._lock:
            for a in range(self.n_arms):
                theta = self.A_inv[a] @ self.b[a]
                # UCB Score using Sherman-Morrison pre-calculated Inverse
                term1 = (theta.T @ x).item()
                term2 = self.alpha * np.sqrt((x.T @ self.A_inv[a] @ x).item())
                p[a] = term1 + term2

        return int(np.argmax(p))

    def update(self, arm: int, features: np.ndarray, reward: float) -> None:
        """
        Updates agent state using Sherman-Morrison for O^2 incremental inverse update.
        """
        if isinstance(features, list):
            features = np.array(features)

        self.scaler.update(features)
        x = self.scaler.transform(features).reshape(-1, 1)

        with self._lock:
            # 1. Sherman-Morrison Incremental Inverse Update (Avoids O^3 Inversion)
            inv_a = self.A_inv[arm]
            denom = self.gamma + (x.T @ inv_a @ x)
            self.A_inv[arm] = (1.0 / self.gamma) * (
                inv_a - (inv_a @ x @ x.T @ inv_a) / denom
            )

            # 2. Update b vector with decay
            self.b[arm] = self.gamma * self.b[arm] + (reward * x)

        if self.use_redis:
            self._sync_to_redis(arm)

    def get_weights(self) -> np.ndarray:
        with self._lock:
            weights = []
            for a in range(self.n_arms):
                theta = self.A_inv[a] @ self.b[a]
                weights.append(theta.flatten())
            return np.array(weights)
