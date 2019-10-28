#!/bin/python3
"""
This script calculates the estimated error of the Totally Positive Kernel
Density Estimator (TPKDE).

Comparison to the regular Kernel Density Estimator (KDE) is also done.
"""

import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from numpy.linalg import inv, cholesky
from scipy.sparse.linalg import arpack
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.datasets import make_spd_matrix
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import multiprocessing as mp
from joblib import Parallel, delayed
print(f"CPU core count: {mp.cpu_count()}")

import tpkde_lib as lib
from tpkde_lib import (
        VERBOSE, convolve_gaussians_with_set, mmc_classic, gpu_mm_hyper as mmc
)


def isPSD(A, tol = 1e-8):
    # return the ends of spectrum of A
    tf = lambda x: arpack.eigsh(x, k=2, which='BE')
    f = arpack.eigh if A.shape[1] == 2 else tf
    vals, vecs = f(A)
    return np.all(vals > -tol)

def bandwidth_sel(x, d, seltype):
    """
    Bandwidth selection using Scott's rule or Silvermann's rule
    """
    if seltype == "scott":
        return (x * (d + 2) / 4.0)**(-1.0 / (d + 4))
    elif seltype == "silvermann":
        return x**(-1./(d+4))
    else:
        assert(False)

def get_gaussian_mean_and_M_covariance(dim):
    """
    Return a random real vector of dimension dim to be used as the mean of
    a Gaussian distribution, along with a covariance matrix whose inverse is an
    M Matrix.
    """
    factor = 1

    # pick a d-dimensional mean uniformly
    mu  = np.random.randn(dim)
    
    # Get an M-matrix of shape (dim, dim)
    sigma = None
    is_PSD = False
    t = 0
    while not is_PSD:
        t += 1
        diag = np.random.randn(dim)
        off_diag = -1 * abs(np.random.randn(int(dim*(dim-1)/2)))
        diag *= factor
        off_diag *= factor
        K = np.zeros((dim, dim))
        k = 0
        for i in range(dim):
            K[i][i] = diag[i]
            for j in range(i):
                K[i][j] = off_diag[k]
                K[j][i] = off_diag[k]
                k += 1
        # invert the M-matrix for covariance matrtix.
        sigma = inv(K)
        is_PSD = isPSD(sigma)
    
    print(f"Needed {t} tries to get cov matrix")
    
    assert(not np.any(sigma <= 0))

    return mu, sigma

def compare_tpkde_and_kde(n, d, n_test):
    """
    Compares TPKDE and KDE.

    params:
    d    :  the dimensionality of the points
    n    :  the number of sample points
    n_test :  the number of test points
    """
    rule = "silvermann"
    
    mu, sig = get_gaussian_mean_and_M_covariance(d) 
    Z = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    MM_Z = mmc(Z)
    m = MM_Z.shape[0]
    print(f"d={Z.shape[1]}, n={Z.shape[0]} -> |MM(z)|={m}")

    real_mean = mu
    sample_mean = np.sum(Z, axis=0) / n
    mmz_mean = np.sum(MM_Z, axis=0) / m

    l2_diff = lambda x, y : np.sqrt( np.sum( (x-y)**2) )
    real_err = l2_diff(real_mean, sample_mean)
    mod_err = l2_diff(real_mean, mmz_mean)
    print(real_err, mod_err)

    return np.array([real_err, mod_err])

    # sample test data from the original dist
    X_test = np.random.multivariate_normal(mean=mu, cov=sig, size=n_test)
    
    h_kde = bandwidth_sel(n, d, rule)
    h_tpkde = bandwidth_sel(m, d, rule)

    def process_i_test_exmple(i):

        x = X_test[i]

        # true PDF
        f_x = multivariate_normal.pdf(x, mean=mu, cov=sig) 
        
        # KDE PDF
        f_hat_x = convolve_gaussians_with_set(x, Z, h_kde)
        
        # TPKDE PDF
        f_hat_x_mod = convolve_gaussians_with_set(x, MM_Z, h_tpkde)

        e =  ((f_x - f_hat_x    ) / f_x)**2
        em = ((f_x - f_hat_x_mod) / f_x)**2
        
        return [e, em]
    
    L = Parallel(n_jobs=mp.cpu_count())(
            delayed(process_i_test_exmple)(i) for i in range(n_test))
    L = np.array(L)
    assert(L.shape == (n_test, 2))
    e = np.sum(L, axis=0)

    #e = np.sqrt(e)
    abs_e = e[0]/n_test
    abs_e_m = e[1]/n_test

    print("(abs) KDE: ", abs_e, " | TPKDE: ", abs_e_m)
    return np.array([abs_e, abs_e_m])


def main():
    d = 2
    ns = [i for i in range(2,300,1)]
    
    errs = []
    for n in ns:
        runs = 1
        err = np.array([0.0, 0.0])
        for i in range(runs):
            err += compare_tpkde_and_kde(n, d, n*runs*2)
        err = err/runs
        errs.append(err)

    errs_KDE = [e[0] for e in errs] 
    errs_TPKDE = [e[1] for e in errs] 
    plt.plot(ns, errs_KDE, "-r", label="KDE")
    plt.plot(ns, errs_TPKDE, "-b", label="TPKDE")
    plt.xlabel("n")
    plt.ylabel("estimated root mean squared error")
    plt.title(f"Estimated root mean squared error: TPKDE vs KDE, d={d}")
    plt.legend()
    plt.show()
    
    return
    ratio_err = [e[0]/e[1] for e in errs] 
    plt.plot(ns, ratio_err, label="ratio of abs error", color='red')
    plt.xlabel("n")
    plt.ylabel("Ratio of Error(KDE)/Error(TPKDE)")
    plt.title("Estimated expected error ratio for d = 2 fixed")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

