import numpy as np

class LinUCB:
    """
    Implementation of the LinUCB Contextual Bandit algorithm.
    As described in Section 3.3 of the research paper.
    """
    def __init__(self, n_arms=3, n_features=5, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # A: Identity matrix for each arm (covariance matrix)
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        # b: Zero vector for each arm (cumulative reward)
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
    def select_arm(self, context, step=0):
        """
        Selects the best arm using the Upper Confidence Bound formula.
        """
        x = np.array(context).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Calculate the inverse of A_a
            A_inv = np.linalg.inv(self.A[a])
            # Estimate theta (regression coefficients)
            theta_hat = A_inv.dot(self.b[a])
            
            # UCB Formula: Mean reward + alpha * standard deviation (confidence interval)
            mean_reward = theta_hat.T.dot(x)
            uncertainty = self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))
            
            p[a] = (mean_reward + uncertainty).item()
            
        return np.argmax(p)
        
    def update(self, arm, context, reward):
        """
        Update the covariance matrix A and the reward vector b for the chosen arm.
        """
        x = np.array(context).reshape(-1, 1)
        self.A[arm] += x.dot(x.T)
        self.b[arm] += reward * x
