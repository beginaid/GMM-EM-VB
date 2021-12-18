import numpy as np
import sys
import fire
import GMMEM

def createData(N):
    """Create 3-dimensional data that is divided into 4 clusters.

    Args:
        N (int): The number of data.

    Returns:
        X (numpy ndarray): Input data whose size is (N, 3).        

    """
    # Mean values
    Mu1 = [5, 5, 5]
    Mu2 = [-5, -5, -5]
    Mu3 = [-5, 5, 5]
    Mu4 = [5, -5, -5]
    
    # Variance-covariance matrices
    Sigma1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Sigma2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Sigma3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Sigma4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Generate data from Gaussian distribution
    X1 = np.random.multivariate_normal(Mu1, Sigma1, N)
    X2 = np.random.multivariate_normal(Mu2, Sigma2, N)
    X3 = np.random.multivariate_normal(Mu3, Sigma3, N)
    X4 = np.random.multivariate_normal(Mu4, Sigma4, N)   

    return np.concatenate([X1, X2, X3, X4])

def main(K, alg):
    """Execute EM algorithm or variational bayes.

    Args:
        K (int): The number of clusters you want.
        iter_max (int): Maximum number of updates.
        thr (float): Threshold of convergence condition.
        alg (string): "EM" or "VB".

    Returns:
        None.

    Note:
        The execute method of each model includes visualization function, and therefore you can get the result of clustering as an image.

    """
    # Set the parameters of those algorithm
    iter_max = 100
    thr = 0.01
    # Create data whose size is 10000
    X = createData(10000)
    # Instantiate the model
    if alg == "EM":
        model = GMMEM.main(K=K)
    else:
        sys.stderr.write("Please specify the argument alg as EM or VB.")
    # Execute the algorithm
    model.execute(X, iter_max=iter_max, thr=thr)

if __name__ == "__main__":
    fire.Fire(main)