import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as la

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
        self.Mu = np.random.randn(self.K, self.D)
        # Set the init variance-covariance matrix as an identity matrix
        self.Sigma = np.array([np.eye(self.D) for i in range(self.K)])
        # Init weight of the mixed Gaussian distribution is generated from Uniform distribution
        self.Pi = np.array([1/self.K for i in range(self.K)])
        # Responsibility of the input data is generate from standard Gaussian distribution
        self.r = np.random.randn(self.N, self.K)

    def e_step(self, X):
        """Execute the E-step of EM algorithm.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.

        Note:
            The following parameters are optimized in this method.
                self.r (numpy ndarray): Responsibility of each data whose size is (N, D).

        """
        # Calculate the responsibilities
        mixed_gaussian_values = self.mix_gauss(X)
        # Update the responsibilities
        self.r = mixed_gaussian_values / np.sum(mixed_gaussian_values, 0)

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
        N_k = np.sum(self.r, 1)[:,None]
        # Calculate and update the optimized Mu
        self.Mu = (self.r @ X) / N_k
        # Calculate the optimized Sigma
        sigma_list = np.zeros((self.N, self.K, self.D, self.D))
        for k in range(self.K):
            for n in range(self.N):
                sigma_list[n][k] = self.r[k][n] * (X[n] - self.Mu[k])[:,None] @ ((X[n] - self.Mu[k]))[None,:]
        # Update the optimized Sigma
        self.Sigma = np.sum(sigma_list, 0) / N_k[:,None]
        # Calculate and update the optimized Pi
        self.Pi = N_k / self.N

    def calc(self, x, mu, sigma):
        """Calculate the value of the D-dimensional Gaussian distribution at each data.

        Args:
            x (numpy ndarray): Input data whose size is D.
            mu (numpy ndarray): Mean value of the D-dimensional Gaussian distribution whose size is D.
            sigma (numpy ndarray): Variance-covariance matrix of the D-dimensional Gaussian distribution whose size is (D, D).

        Returns:
            gaussian_value (numpy ndarray): Value of the D-dimensional Gaussian distribution at each data whose size is D.

        """
        # Calculate exponent of D-dimensional Gaussian distribution
        exp = -0.5 * (x - mu).T @ la.inv(sigma).T @ (x - mu)
        # Calculate denomin of D-dimensional Gaussian distribution
        denomin = np.sqrt(la.det(sigma)) * (np.sqrt(2*np.pi) ** self.D)
        gaussian_value = np.exp(exp) / denomin
        return gaussian_value
    
    def gauss(self, X, mu, sigma):
        """Calculate the values of the D-dimensional Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).
            mu (numpy ndarray): Mean value of the D-dimensional Gaussian distribution whose size is D.
            sigma (numpy ndarray): Variance-covariance matrix of the D-dimensional Gaussian distribution whose size is (D, D).

        Returns:
            gaussian_values (numpy ndarray): Values of the D-dimensional Gaussian distribution at N data whose size is (N, D).

        """
        gaussian_values = np.array([self.calc(X[i], mu, sigma) for i in range(self.N)])
        return gaussian_values
    
    def mix_gauss(self, X):
        """Calculate the values of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            mixed_gaussian_values (numpy ndarray): Values of the 1-dimensional Gaussian distribution at N data whose size is (N, D).

        """
        mixed_gaussian_values = np.array([self.Pi[i] * self.gauss(X, self.Mu[i], self.Sigma[i]) for i in range(self.K)])
        return mixed_gaussian_values

    def log_likelihood(self, X):
        """Calculate the log-likelihood of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            log_likelihood (numpy ndarray): Values of the 1-dimensional Gaussian distribution at N data whose size is (N, D).

        """
        mixed_gaussian_values = self.mix_gauss(X)
        log_likelihood = np.sum([np.log(np.sum(mixed_gaussian_values, 0)[n]) for n in range(self.N)])
        return log_likelihood

    def classify(self, X):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            labels (numpy ndarray): Maximum values of the responsibility at each data whose size is N.

        """
        labels = np.argmax(self.r, 0)
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
        # If the number of cluster is less than or equal to 8, use the custome color list.
        if self.K <= 8:
            color_list = ["#ec442c", "#067606", "#2474bc", "#940c7c", "#f27b6a", "#08a608", "#3a8ed9", "#c310a3"]         
        # Otherwise, use the tab10 color map
        else:
            cm = plt.get_cmap("tab10")   
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Change the perspective
        ax.view_init(elev=10, azim=70)
        # Visualize each clusters
        for k in range(self.K):
            cluster_indexes = np.where(labels==k)[0]
            if self.K <= 8:
                ax.plot(X[cluster_indexes, 0], X[cluster_indexes, 1], X[cluster_indexes, 2], "o", ms=0.5, color=color_list[k])
            else:
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
        log_likelihood_list = np.append(log_likelihood_list, self.log_likelihood(X))

        # Start the iteration
        for i in range(iter_max):
            # Execute E-step
            self.e_step(X)
            # Execute M-step
            self.m_step(X)
            # Add the current log-likelihood
            log_likelihood_list = np.append(log_likelihood_list, self.log_likelihood(X))
            # Print the gap between the previous log-likelihood and current one
            print("Previous log-likelihood gap: " + str(np.abs(log_likelihood_list[i] - log_likelihood_list[i+1])))
            # Visualization is performed when the convergence condition is met or when the upper limit is reached
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i+1]) < thr) or (i == iter_max - 1):
                self.visualize(X)
                break
