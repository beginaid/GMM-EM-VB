from collections import Counter

import fire
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


class GMMEM:
    def __init__(self, K):
        """Constructor.

        Args:
            K (int): The number of clusters.

        Returns:
            None.

        Note:
            eps (float): Small amounts to prevent overflow and underflow.
        """
        self.K = K
        self.eps = np.spacing(1)

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
        self.sigma = np.tile(np.eye(self.D), (self.K, 1, 1))
        # Init weight of the mixed Gaussian distribution is generated from Uniform distribution
        self.pi = np.ones(self.K) / self.K
        # Responsibility of the input data is generate from standard Gaussian distribution
        # It will be updated at E-step, and this initialization practically has no meaning
        self.r = np.random.randn(self.N, self.K)

    def gmm_pdf(self, X):
        """Calculate the log-likelihood of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            Probability density function (numpy ndarray): Values of the mixed D-dimensional Gaussian distribution at N data whose size is (N, K).
        """
        return np.array(
            [
                self.pi[k]
                * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k])
                for k in range(self.K)
            ]
        ).T  # (N, K)

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
        gmm_pdf = self.gmm_pdf(X)  # (N, K)
        log_r = np.log(gmm_pdf) - np.log(
            np.sum(gmm_pdf, 1, keepdims=True) + self.eps
        )  # (N, K)
        # Modify to exponential-domain
        r = np.exp(log_r)  # (N, K)
        # Replace the element where nan appears
        r[np.isnan(r)] = 1.0 / (self.K)  # (N, K)
        # Update the optimized r
        self.r = r  # (N, K)

    def m_step(self, X):
        """Execute the M-step of EM algorithm.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.

        Note:
            The following parameters are optimized in this method.
                self.mu (numpy ndarray): Mean vector of the mixed Gaussian distribution whose size is (K, D).
                self.sigma (numpy ndarray): Variance-covariance matrix of the mixed Gaussian distribution whose size is (K, D, D).
                self.pi (numpy ndarray): Weight of the mixed Gaussian distribution whose size is K.
        """
        # Calculate the parameters (Mu, Sigma, Pi) that maximize the Q function
        # Prepare N_k to calculate the parameters
        N_k = np.sum(self.r, 0)  # (K)
        # Calculate and update the optimized Pi
        self.pi = N_k / self.N  # (K)
        # Calculate and update the optimized Mu
        self.mu = (self.r.T @ X) / (N_k[:, None] + self.eps)  # (K, D)
        # Calculate and update the optimized Sigma
        r_tile = np.tile(self.r[:, :, None], (1, 1, self.D)).transpose(
            1, 2, 0
        )  # (K, D, N)
        res_error = np.tile(X[:, :, None], (1, 1, self.K)).transpose(2, 1, 0) - np.tile(
            self.mu[:, :, None], (1, 1, self.N)
        )  # (K, D, N)
        self.sigma = ((r_tile * res_error) @ res_error.transpose(0, 2, 1)) / (
            N_k[:, None, None] + self.eps
        )  # (K, D, D)

    def visualize(self, X):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Execute classification
        labels = np.argmax(self.r, 1)  # (N)
        # Visualize each clusters
        label_frequency_desc = [l[0] for l in Counter(labels).most_common()]  # (K)
        # Prepare the visualization
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = Axes3D(fig)
        fig.add_axes(ax)
        cm = plt.get_cmap("tab10")
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Change the perspective
        ax.view_init(elev=10, azim=70)
        for k in range(len(label_frequency_desc)):
            cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
            ax.plot(
                X[cluster_indexes, 0],
                X[cluster_indexes, 1],
                X[cluster_indexes, 2],
                "o",
                ms=0.5,
                color=cm(k),
            )
        plt.savefig("./results/clusters.png")
        plt.close(fig)

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
        log_likelihood_list = []
        # Calculate the initial log-likelihood
        log_likelihood_list.append(
            np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
        )
        # Start the iteration
        for i in range(iter_max):
            # Execute E-step
            self.e_step(X)
            # Execute M-step
            self.m_step(X)
            # Add the current log-likelihood
            log_likelihood_list.append(
                np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
            )
            # Print the gap between the previous log-likelihood and current one
            print(
                "Log-likelihood gap: "
                + str(
                    round(
                        np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]), 2
                    )
                )
            )
            # Visualization is performed when the convergence condition is met or when the upper limit is reached
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
                i == iter_max - 1
            ):
                print(f"EM algorithm has stopped after {i + 1} iteraions.")
                self.visualize(X)
                break


def main(K):
    return GMMEM(K)


if __name__ == "__main__":
    fire.Fire(main)
