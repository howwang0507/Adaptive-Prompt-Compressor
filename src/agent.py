import numpy as np

class OnlineScaler:
    """
    Implements Welford's algorithm for online mean and variance estimation.
    Ensures that features with different scales (e.g. length vs codeness) 
    don't dominate the covariance matrix.
    """
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / (self.n - 1) if self.n > 1 else np.ones_like(self.mean)

    def transform(self, x):
        if self.n < 2:
            return x
        std = np.sqrt(self.var)
        std[std == 0] = 1.0 # Prevent division by zero
        return (x - self.mean) / std

class LinUCB:
    """
    Contextual Multi-Armed Bandit using the LinUCB algorithm.
    Optimized with Online Scaling and Forgetting Factor (Concept Drift).
    """
    def __init__(self, n_arms, n_features, alpha=1.0, gamma=0.99):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma # Forgetting factor
        
        # A_a = Identity matrix (initial covariance)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        # b_a = Zero vector (initial reward accumulation)
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
        # Pre-computed inverse for speed (A_inv = A^-1)
        self.A_inv = [np.identity(n_features) for _ in range(n_arms)]
        
        # Online scaling to handle feature scale disparity
        self.scaler = OnlineScaler(n_features)

    def select_arm(self, features, step=0):
        # Apply Online Scaling
        x = self.scaler.transform(features).reshape(-1, 1)
        
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            # UCB Score = Expected Reward + Uncertainty Bonus
            p[a] = theta.T @ x + self.alpha * np.sqrt(x.T @ self.A_inv[a] @ x)
            
        return np.argmax(p)

    def update(self, arm, features, reward):
        # Update Scaling Stats
        self.scaler.update(features)
        x = self.scaler.transform(features).reshape(-1, 1)
        
        # A_a = gamma * A_a + x * x.T (With Forgetting Factor)
        # Note: Initial identity is not decayed to maintain regularization stability
        self.A[arm] = self.gamma * self.A[arm] + (x @ x.T)
        self.b[arm] = self.gamma * self.b[arm] + (reward * x)
        
        # Update Pre-computed Inverse (Incremental Woodbury update could be used, but d=5 is small)
        self.A_inv[arm] = np.linalg.inv(self.A[arm])

    def get_weights(self):
        """
        Returns the theta weights for each arm. 
        Crucial for interpretability (Section 5.3 of the paper).
        """
        weights = []
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            weights.append(theta.flatten())
        return np.array(weights)
