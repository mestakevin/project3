import numpy as np
import emcee
import matplotlib.pyplot as plt

a0 = 5.29e-11

##-----------------------------------------------------------##
def radial_func(r):
    normalization = 1 / (81 * np.sqrt(3 * a0**3))
    return normalization * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))
##-----------------------------------------------------------##
def radial_prob_func(r):
    # Make sure r is treated as a scalar
    r = r[0] if isinstance(r, list) else r
    return (radial_func(r) ** 2) * r ** 2
##-----------------------------------------------------------##
def log_prob(r):
    if r[0] < 0:
        return -np.inf
    return np.log(radial_prob_func(r[0]) + 1e-100)
##-----------------------------------------------------------##
