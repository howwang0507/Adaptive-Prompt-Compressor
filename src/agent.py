import numpy as np

class LinUCB:
    def __init__(self, n_arms=3, n_features=5, alpha=1.0):
        self.n_arms, self.n_features, self.alpha = n_arms, n_features, alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
    def select_arm(self, context, step=0):
        x = np.array(context).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            p[a] = (A_inv.dot(self.b[a]).T.dot(x) + self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))).item()
        return np.argmax(p)
        
    def update(self, arm, context, reward):
        x = np.array(context).reshape(-1, 1)
        self.A[arm] += x.dot(x.T)
        self.b[arm] += reward * x
