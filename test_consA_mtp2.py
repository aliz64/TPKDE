#!/bin/python3

"""
Test whether an MTP_2 distribution satisfies Constraint A.
We already know the uniform distrubution over a min-max closed set does this.
The hypothesis we are testing is a stronger one.
"""

import numpy as np

import main as m


def test_constraint_A_at(p1, p2, mu, sig):
  # Note: only test at identity permutation (see paper)
  # Since condition needs to hold for ALL permutations, a counterexample
  # in one should be sufficient.
  d = p1.shape[0]
  
  x_1_0 = min(p1[0], p2[0])
  x_1_1 = max(p1[0], p2[0])
  
  x_2_0 = min(p1[1], p2[1])
  x_2_1 = max(p1[1], p2[1])

  
  S = 0
  for i in range(2**(d-2)):
    a = f"{i:0{d-2}b}" if d > 2 else ""
    x_i_0 = [];
    x_i_1 = [];
    for bit in range(len(a)):
      if a[bit] == "0":
        x_i_0.append(p1[2+bit])
        x_i_1.append(p2[2+bit])
      elif a[bit] == "1":
        x_i_0.append(p2[2+bit])
        x_i_1.append(p1[2+bit])
      else:
        assert(False)

    f_x_i_0 = np.array([x_1_0, x_2_0] + x_i_0) # min, min
    f_x_i_1 = np.array([x_1_1, x_2_1] + x_i_1) # max, max

    op_0 =  np.array([x_1_1, x_2_0] + x_i_0) # max, min
    op_1 =  np.array([x_1_0, x_2_1] + x_i_1) # min, max
    
    fn = lambda x : m.multivariate_normal.pdf(x, mean=mu, cov=sig)
    termP = fn(f_x_i_0) * fn(f_x_i_1)
    termN = fn(op_0) * fn(op_1)

    S = S + termP - termN
  
  res = S >= 0
  return res

def main():
  n_pairs = 100
  suite = {2:5, 3:10, 4:100}
  for d in suite:
    v = suite[d] 
    for vv in range(v):
      mu, sig = m.get_gaussian_mean_and_M_covariance(d)
      ans = True
      for i in range(n_pairs):
        # Sample two points - just use the acrtual distribution
        x_i_0, x_i_1 = m.np.random.multivariate_normal(
            mean=mu, cov=sig, size=2)
        ans = ans and test_constraint_A_at(x_i_0, x_i_1, mu, sig)
      add = "" if ans else " <------- FALSE HERE"
      print("#"*10 + f" d = {d} " + "#"*5 + f" --> {ans} {add}")
    
if __name__ == "__main__":
  main()
