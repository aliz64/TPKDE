#!/usr/bin/env python3
"""
Test various condition violations.

1) Testing whether any MTP_2 Gaussian satisfies Constraint A.

2) Test whether the uniform distribution over a min-max closed set, convolved
   with a Gaussian that is not a scaled standard Gaussian, produces an MTP_2
   distribution. This relates to Counterexample 5 and Appendix B.
"""

from argparse import ArgumentParser
import numpy as np

from main import (
    multivariate_normal,
    get_gaussian_mean_and_M_covariance
)

def test_constraint_a_at(p_1, p_2, mean, sigma):
    # pylint: disable=too-many-locals
    """
    Helper function to partially test constraint A at the points p1 and p2
    (in R^d) for the Gaussian distribution with mean  in R^d and covariance
    = sigma in R^(dxd).
    The actual constraint requires this to be true at all pairs of  points in
    R^d.

    Note: only test at identity permutation (see paper for details). Since
          condition needs to hold for ALL permutations, a counterexample in one
          should be sufficient.
    """
    assert p_1.shape == p_2.shape and p_2.shape == mean.shape
    dim = p_1.shape[0]
    assert sigma.shape == (dim, dim)

    x_1_0 = min(p_1[0], p_2[0])
    x_2_0 = min(p_1[1], p_2[1])

    x_1_1 = max(p_1[0], p_2[0])
    x_2_1 = max(p_1[1], p_2[1])

    current_sum = 0
    for i in range(2**(dim - 2)):
        # Format i into a binary representation with d-2 zeros. This is OK since
        # in the loop, i is max 2^(d-2) - 1
        bits = f"{i:0{dim - 2}b}" if dim > 2 else ""
        x_i_0 = []
        x_i_1 = []
        for bit, _ in enumerate(bits):
            if bits[bit] == "0":
                x_i_0.append(p_1[2+bit])
                x_i_1.append(p_2[2+bit])
            elif bits[bit] == "1":
                x_i_0.append(p_2[2+bit])
                x_i_1.append(p_1[2+bit])
            else:
                assert False

        f_x_i_0 = np.array([x_1_0, x_2_0] + x_i_0) # min, min
        f_x_i_1 = np.array([x_1_1, x_2_1] + x_i_1) # max, max

        op_0 =  np.array([x_1_1, x_2_0] + x_i_0) # max, min
        op_1 =  np.array([x_1_0, x_2_1] + x_i_1) # min, max

        dist = lambda x : multivariate_normal.pdf(x, mean=mean, cov=sigma)
        term_positive = dist(f_x_i_0) * dist(f_x_i_1)
        term_negative = dist(op_0) * dist(op_1)
        current_sum = current_sum + term_positive - term_negative

    return current_sum >= 0

def test_constraint_a_for_mtp_2_gaussians():
    """
    Test whether any Gaussian MTP_2 distribution satisfies Constraint A.

    We already have proved the uniform distribution over any finite min-max
    closed set satisfies Constraint A.

    TODO: We know that the convolution of two MTP_2 Gaussians is not necessarily
    MTP_2. But the convolution of a scaled standard Gaussian (which is MTP_2)
    with a Constraint A satisfying distribution is MTP_2. So if any Gaussian
    MTP_2 distribution satisfies Constraint A, then the convolution of any MTP_2
    Gaussian with a standard Gaussian would be MTP_2. This may also be provable
    by considering the sum of the inverse of an M-matrix and a scaled identity
    matrix, and showing inverse of the result is an M-matrix (the convolution of
    two Gaussians results in a Gaussian with the covariance matrices summed).
    """

    # The test is done by getting an arbitraty MTP_2 Gaussian and sampling two
    # points from it (although any two points can be used in reality). Then we
    # check the Constraint A condition with respect to the density.

    # number of pairs to test on
    n_pairs_to_test = 100
    # pairs of (dimension, number of trials)
    test_suite = {2:50, 3:50, 4:50}

    for dim in test_suite:
        n_trials = test_suite[dim]
        for trial in range(n_trials):
            mu, sigma = get_gaussian_mean_and_M_covariance(dim)

            print(f"d={dim}, trial={trial}, pairs={n_pairs_to_test}")

            for _ in range(n_pairs_to_test):
                # Sample two points - just use the actual distribution, although
                # any two points in R^d can work.
                x_0, x_1 = np.random.multivariate_normal(mean=mu,
                                                             cov=sigma,
                                                             size=2)

                if not test_constraint_a_at(x_0, x_1, mu, sigma):
                    print("mu: ", mu)
                    print("sigma: ", sigma)
                    print("x_0: ", x_0)
                    print("x_1: ", x_1)
                    assert False, "condition failed!"
                    return

if __name__ == "__main__":
    test_constraint_a_for_mtp_2_gaussians()
