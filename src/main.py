import numpy as np
import os
import sys
import fire
import GMMEM
import GMMVB

def createData():
    """Create 3-dimensional data that is divided into 4 clusters.

    Args:
        None.

    Returns:
        X (numpy ndarray): Input data whose size is (N, 3).        

    """
    # The number of data at each cluster
    N1 = 4000
    N2 = 3000
    N3 = 2000
    N4 = 1000

    # Mean values
    Mu1 = [5, -5, -5]
    Mu2 = [-5, 5, 5]
    Mu3 = [-5, -5, -5]
    Mu4 = [5, 5, 5]    

    # Variance-covariance matrices
    Sigma1 = [[1, 0, -0.25], [0, 1, 0], [-0.25, 0, 1]]
    Sigma2 = [[1, 0, 0], [0, 1, -0.25], [0, -0.25, 1]]
    Sigma3 = [[1, 0.25, 0], [0.25, 1, 0], [0, 0, 1]]
    Sigma4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Generate data from Gaussian distribution
    X1 = np.random.multivariate_normal(Mu1, Sigma1, N1)
    X2 = np.random.multivariate_normal(Mu2, Sigma2, N2)
    X3 = np.random.multivariate_normal(Mu3, Sigma3, N3)
    X4 = np.random.multivariate_normal(Mu4, Sigma4, N4)   

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
    # Create results directory to output images
    if not os.path.isdir("results"):
        os.makedirs("results")
    # Set the parameters of those algorithm
    iter_max = 100
    thr = 0.001
    # Create data at each cluster whose size is 2500, resulting in 10000 data overall
    X = createData()
    # Instantiate the model
    if alg == "EM":
        model = GMMEM.main(K=K)
    elif alg == "VB":
        model = GMMVB.main(K=K)
    else:
        sys.stderr.write("Please specify the argument alg as EM or VB.")
    # Execute the algorithm
    model.execute(X, iter_max=iter_max, thr=thr)

if __name__ == "__main__":
    fire.Fire(main)