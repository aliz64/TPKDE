"""
Library of methods needed for TPKDE computation and evaluation.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

from numpy.linalg import inv
from scipy.stats import norm
from scipy.stats import multivariate_normal

"""
Currently there are 4 different min-max closure methods/algorithms implemented.

1) mmc_classic : Naive CPU. Slow as expected in this exponential regime. 
2) gpu_mm    : GPU algorithm parallelizing over current pairs of points.
3) gpu_mm_grid : GPU algorithm parallelizing over all possible points.
4) gpu_mm_hyper: GPU algorithm parallelizing over current points.

See python code in 'gpu_impls_mmz.py' and CUDA code in 'cuda_code_strings.py'.
"""
from gpu_impls_mmz import (
    VERBOSE, gpu_mm, gpu_mm_grid, gpu_mm_hyper, mmc_classic
)

GREEN = '\033[92m'
CYAN = '\033[96m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def convolve_gaussians_with_set(x, MM_Z, h=0.2):
    """
    Calculate probability density 'f' at point 'x', where 'f' is the
    (normalized) sum of i.i.d. Gaussian distributions centered at the elements
    of the finite real vector set 'MM_Z'.
    
    x  : A point in this space, at which we want to calculate this density.
    MM_Z : The set of points at which the Gaussians are centered.
             Given as a numpy array.
    h  : h*I is the covariance matrix of the Gaussians. Here 'I' lies in
             R^(d x d).
    """

    n, d = MM_Z.shape

    # Subtract 'x' from all points in the set.
    # Calculate zero-centered Gaussian PDF at these modified points.
    # This works since a Gaussian centered at (x,y), evaluated at (a,b) is the
    # same as a Gaussian centered at (0,0) evaluated at (x-a,y-b).
    MM_Z = MM_Z - x
    p_v = multivariate_normal.pdf(MM_Z, mean=np.zeros(d), cov=np.identity(d)*h)
    return sum(p_v)/n

def run_and_time(alg_f, X_sample, iters=float('inf')):
    """
    Runs and times an MM(Z) algorithm 'alg_f' on X_sample, for 'iters'
    iterations.
    """
    N, D = X_sample.shape
    
    print(CYAN + f"Running and timing algorithm: {alg_f.__name__}" + ENDC)
    if iters != float('inf'):
        print(CYAN + f"Running for fixed number {iters} of iterations." + ENDC)
    
    start_time = time.time()
    answer = alg_f(X_sample, iters)
    taken_time = time.time() - start_time
    
    print(CYAN + "{}:  d={}, n={} -> |MM(Z)|={}  t={} seconds".format(
            alg_f.__name__, D, N, len(answer), round(taken_time, 4)) + ENDC)

    return answer, taken_time

def plot_mmz_against_n_d(alg_f=mmc_classic, dnlist={2:[10,15,20]}):
    """
    For each given dimension and the coressponding sample size N, run the given
    algorithm and plot the size of MM(Z) using a randomly generated sample.
    """
    plt.figure()
    for d in dnlist:
        ns = []
        MM_lens  = []
        for n in dnlist[d]:
            X_sample = np.random.rand(n, d)

            a, t = run_and_time(alg_f, X_sample)
            
            # mmc_classic returns a list of tuples rather than a D-array
            if alg_f.__name__ == "mmc_classic":
                s = len(a)
            else:
                s = a.shape[0]

            ns.append(n)
            MM_lens.append(s)
            print("d={}, n={} -> |MM(z)|={}".format(d, n, s))

        plt.plot(ns, MM_lens, label="d={}".format(d))

    plt.xlabel('n')
    plt.ylabel('|MM(Z)|')
    plt.title("|MM(Z)| vs. n with d varying")
    plt.legend()
    plt.savefig("MMZ_size_vs_n_for_d.pdf")

def alg_time_comparison(alg_list, dnlist={3:[10,15,20]}, no_comp=False):
    """
    Compares the algorithms in alg_list on given dimensions and sample sizes.
    Plots the relative speeds (using first algorithm as base reference).
    Separate plot is made for each dimension.
    """
    assert len(alg_list) >= 2
    for d in dnlist:
        ns = []
        time_lists = [[] for alg in alg_list]

        for n in dnlist[d]:
            print(f"Running for: d={d}, n={n}")
            X_sample = np.random.randn(n, d)
            
            # Reference answer and time
            a, t = run_and_time(alg_list[0], X_sample)
            time_lists[0].append(1 if not no_comp else t)
            l = len(a) if alg_list[0].__name__ == "mmc_classic" else a.shape[0]

            for i in range(1, len(alg_list)):
                # answer amd time for i-th algorithm
                ai, ti = run_and_time(alg_list[i], X_sample)
                assert ai.shape[0] == l
                plotval = t/ti if not no_comp else ti
                time_lists[i].append(plotval)
                print(
                    GREEN + f"For d={d}, n={n}, the speedup is {plotval}x" + "\n" + ENDC)

            ns.append(n)

        n_groups = len(dnlist[d])
        index = np.arange(n_groups)
        bar_width = 1.0/len(alg_list)
        ns = np.array(ns)

        name_map = {
        "mmc_classic":"Algorithm 1",
        "gpu_mm":"Algorithm 2",
        #"gpu_mm_grid":"Algorithm 3",
        "gpu_mm_hyper":"Algorithm 3"}

        plt.figure()
        for i, alg in enumerate(alg_list):
            plt.bar(ns + bar_width*i, time_lists[i], bar_width,
                    label=name_map[alg.__name__])

        plt.xlabel("n")
        if not no_comp:
            ylab = f"speedup (ratio) over {name_map[alg_list[0].__name__]}"
        else:
            ylab = "time (ms)"
        plt.ylabel(ylab)
        plt.title(f"speeds vs. n for different algorithms (d = {d})")
        plt.legend()
        plt.savefig(f"time_comparisons_d{d}.pdf")
