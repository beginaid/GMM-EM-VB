import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from collections import Counter
from scipy.special import logsumexp
from mpl_toolkits.mplot3d import Axes3D

class GMMEM():
    def __init__(self, K):
        """Constructor.

        Args:
            K (int): The number of clusters.

        Returns:
            None.

        """
        self.K = K

    def init_params(self, X):
        """Init the parameters.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.

        """
        # The size of X is (N, D)
        self.N, self.D = X.shape
        # Mean value of the Gaussian distribution is generate from standard Gaussian distribution
        self.mu = np.random.randn(self.K, self.D)
        # Set the init variance-covariance matrix as an identity matrix
        self.sigma = np.array([np.eye(self.D) for i in range(self.K)])
        # Init weight of the mixed Gaussian distribution is generated from Uniform distribution
        self.pi = np.array([1 / self.K for i in range(self.K)])
        # Responsibility of the input data is generate from standard Gaussian distribution
        # Responsibility will be updated at E-step, and this initialization practically has no meaning
        self.r = np.random.randn(self.N, self.K)

    def log_likelihood(self, X):
        """Calculate the log-likelihood of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            log_likelihood (numpy ndarray): Values of the D-dimensional Gaussian distribution at N data whose size is (N, K).

        """
        log_pi = np.tile(np.log(self.pi + np.spacing(1))[None, :], (self.N, 1)) # (N, K)
        log_sigma_det = np.tile(np.log(la.det(self.sigma) + np.spacing(1))[None,:], (self.N, 1)) # (N, K)
        res_error = np.tile(X[:,None,None,:], (1, self.K, 1, 1)) - np.tile(self.mu[None,:,None,:], (self.N, 1, 1, 1)) # (N, K, 1, D)
        sigma_inv = np.tile((la.pinv(self.sigma)[None,:,:,:]), (self.N, 1, 1, 1)) # (N, K, D, D)
        log_exponent = np.einsum("nkod, nkde -> nkoe", np.einsum("nkod, nkde -> nkoe", res_error, sigma_inv), res_error.transpose(0,1,3,2))[:,:,0,0] # (N, K)
        log_likelihood = log_pi + (-0.5 * self.D * np.log(2 * np.pi)) + (-0.5 * log_sigma_det) + (-0.5 * log_exponent) # (N, K)
        return log_likelihood # (N, K)

    def e_step(self, X):
        """Execute the E-step of EM algorithm.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.

        Note:
            The following parameters are optimized in this method.
                self.r (numpy ndarray): Responsibility of each data whose size is (N, K).

        """
        # Calculate the responsibilities in log-domain
        log_likelihood = self.log_likelihood(X)
        log_r = log_likelihood - logsumexp(log_likelihood, 1, keepdims=True)
        # Modify to exponential-domain
        r = np.exp(log_r)
        # Replace the element where nan appears
        r[np.isnan(r)] = 1.0 / (self.K)
        self.r = r

    def m_step(self, X):
        """Execute the M-step of EM algorithm.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.

        Note:
            The following parameters are optimized in this method.
                self.Mu (numpy ndarray): Mean vector of the mixed Gaussian distribution.
                self.Sigma (numpy ndarray): Variance-covariance matrix of the mixed Gaussian distribution.                
                self.Pi (numpy ndarray): Weight of the mixed Gaussian distribution whose size is K.

        """
        # Calculate the parameters (Mu, Sigma, Pi) that maximize the Q function
        # Prepare N_k to calculate the parameters
        N_k = np.sum(self.r, 0) # (K)
        # Calculate and update the optimized Mu
        self.mu = (self.r.T @ X) / (N_k[:, None] + np.spacing(1)) # (K, D)
        # Calculate and update the optimized Sigma
        r_d = np.tile(self.r[:,:,None], (1, 1, self.D)).transpose(1, 2, 0) # (K, D, N)
        res_error = np.tile(X[:,:,None], (1, 1, self.K)).transpose(2, 1, 0) - np.tile(self.mu[:,:,None], (1, 1, self.N)) # (K, D, N)
        self.sigma = np.einsum("kdn, kne -> kde", np.einsum("kdn, kdn -> kdn", r_d, res_error), res_error.transpose(0, 2, 1)) / (N_k[:,None,None] + np.spacing(1)) # (K, D, D)
        # Calculate and update the optimized Pi
        self.pi = N_k / self.N # (K)

    def classify(self, X):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            labels (numpy ndarray): Maximum values of the responsibility at each data whose size is N.

        """
        labels = np.argmax(self.r, 1) # (N)
        return labels

    def visualize(self, X):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        
        Note:
            This method executes plt.show, and does not execute plt.close.

        """
        # Execute classification
        labels = self.classify(X)
        # Prepare the visualization
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = Axes3D(fig)
        # Use the tab10 color map
        cm = plt.get_cmap("tab10")   
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Change the perspective
        ax.view_init(elev=10, azim=70)
        # Visualize each clusters
        label_frequency_desc = [l[0] for l in Counter(labels).most_common()]
        for k in range(len(label_frequency_desc)):
            cluster_indexes = np.where(labels==label_frequency_desc[k])[0]
            ax.plot(X[cluster_indexes, 0], X[cluster_indexes, 1], X[cluster_indexes, 2], "o", ms=0.5, color=cm(k))
        plt.show()

    def execute(self, X, iter_max, thr):
        """Execute EM algorithm.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).
            iter_max (int): Maximum number of updates.
            thr (float): Threshold of convergence condition.

        Returns:
            None.

        """
        # Init the parameters
        self.init_params(X)
        # Prepare the list of the log_likelihood at each iteration
        log_likelihood_list = np.array([])
        # Calculate the initial log-likelihood
        log_likelihood_list = np.append(log_likelihood_list, np.mean(self.log_likelihood(X)))

        # Start the iteration
        for i in range(iter_max):
            # Execute E-step
            self.e_step(X)
            # Execute M-step
            self.m_step(X)
            # Add the current log-likelihood
            log_likelihood_list = np.append(log_likelihood_list, np.mean(self.log_likelihood(X)))
            # Print the gap between the previous log-likelihood and current one
            print("Log-likelihood gap: " + str(round(np.abs(log_likelihood_list[i] - log_likelihood_list[i+1]), 2)))
            # Visualization is performed when the convergence condition is met or when the upper limit is reached
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i+1]) < thr) or (i == iter_max - 1):
                print(f"EM algorithm has stopped after {i + 1} iteraion.")
                self.visualize(X)
                break

def main(K):
    return GMMEM(K)

if __name__ == "__main__":
    fire.Fire(main)