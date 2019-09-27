import numpy as np


def main():
    n = 3                # number of dimensions
    N = 5000             # number of samples
    np.random.seed(100)  # ensure fixed seed

    # define the mean to be an N dimensional matrix with 0
    mean = np.zeros(n)
    # define the covariance matrix to be N dimensional identity matrix
    cov = np.identity(n)
    # draw random samples from the distribution
    z = np.random.multivariate_normal(mean, cov, N).T

    # generate random mu
    mu = np.random.rand(n)

    # generate random sigma (positive definite matrix)
    a1 = np.random.rand(n, n)
    sigma = np.dot(a1, a1.transpose())

    # calculate matrix A
    A = np.linalg.cholesky(sigma)

    # calculate B
    B = mu

    # apply transformation Az + B
    x = np.add(np.matmul(A, z).T, B[np.newaxis, :])


if __name__ == '__main__':
    main()
