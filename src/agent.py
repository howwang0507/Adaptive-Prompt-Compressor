import numpy as np
import threading
from typing import List, Tuple, Optional, Any

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
        if self.n < 10: return False # Need warm-up
        z_scores = np.abs((x - self.mean) / (np.sqrt(self.var) + 1e-6))
        return bool(np.any(z_scores > threshold))

class LinUCB:
    """
    Contextual Multi-Armed Bandit using the LinUCB algorithm.
    Enhanced for Numerical Stability, Thread-Safety, and OOD Protection.
    """
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0, 
                 gamma: float = 0.99, lambda_reg: float = 1.0,
                 domain_priors: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None):
        self.n_arms: int = n_arms
        self.n_features: int = n_features
        self.alpha: float = alpha
        self.gamma: float = gamma
        
        # Ridge Regularization or Meta-Learning Warm Start
        if domain_priors:
            # Zero-Shot Domain Adaptation: Load pre-trained covariance and reward matrices
            self.A_inv: List[np.ndarray] = [domain_priors[a][0] for a in range(n_arms)]
            self.b: List[np.ndarray] = [domain_priors[a][1] for a in range(n_arms)]
            print("🚀 Meta-Learning: Initialized with Zero-Shot Domain Priors.")
        else:
            self.A_inv: List[np.ndarray] = [np.identity(n_features) / lambda_reg for _ in range(n_arms)]
            self.b: List[np.ndarray] = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
        # Scaler for online normalization

        self.scaler: OnlineScaler = OnlineScaler(n_features)
        
        # Thread safety lock
        self._lock: threading.Lock = threading.Lock()

    def select_arm(self, features: np.ndarray) -> int:
        """
        Selects the optimal arm using UCB strategy.
        Now performs OOD check to prevent radical failures.
        """
        if self.scaler.is_ood(features):
            # Fallback to Arm 0 (Conservative) for OOD inputs
            return 0
            
        x = self.scaler.transform(features).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        
        with self._lock:
            for a in range(self.n_arms):
                theta = self.A_inv[a] @ self.b[a]
                # UCB Score using Sherman-Morrison pre-calculated Inverse
                p[a] = theta.T @ x + self.alpha * np.sqrt(x.T @ self.A_inv[a] @ x)
            
        return int(np.argmax(p))

    def update(self, arm: int, features: np.ndarray, reward: float) -> None:
        """
        Updates agent state using Sherman-Morrison for O^2 incremental inverse update.
        Ensures Numerical Stability and Thread-Safety.
        """
        self.scaler.update(features)
        x = self.scaler.transform(features).reshape(-1, 1)
        
        with self._lock:
            # 1. Sherman-Morrison Incremental Inverse Update (Avoids O^3 Inversion)
            # A_inv = (1/gamma) * (A_inv - (A_inv * x * x.T * A_inv) / (gamma + x.T * A_inv * x))
            inv_a = self.A_inv[arm]
            denom = self.gamma + (x.T @ inv_a @ x)
            self.A_inv[arm] = (1.0 / self.gamma) * (inv_a - (inv_a @ x @ x.T @ inv_a) / denom)
            
            # 2. Update b vector with decay
            self.b[arm] = self.gamma * self.b[arm] + (reward * x)

    def get_weights(self) -> np.ndarray:
        with self._lock:
            weights = []
            for a in range(self.n_arms):
                theta = self.A_inv[a] @ self.b[a]
                weights.append(theta.flatten())
            return np.array(weights)
