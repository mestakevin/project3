import numpy as np
import tqdm
import numpy.random
import random
from matplotlib import pyplot as plt
from .custom_mcmc import MCMC

numpy.random.seed(12345)
a0 = 5.29e-11  # Bohr radius in meters


def radial_func(r):
    #r = param[0]
    normalization = 1 / (81 * np.sqrt(3 * (np.pi) * a0**3))
    return normalization * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))


def radial_prob_func(r):
    #r = param[0]
    return (radial_func(r) ** 2 )* r **2

def proposal(r):
    new_r = r + random.gauss(0, a0 * 3)
    return new_r

def log_prob(r):
    if r < 0:
        return -np.inf
    return np.log(radial_prob_func(r) + 1e-100)


def convergenceCheck(chains):

    m, n = chains.shape  
    
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    
    
    B = n * np.var(chain_means, ddof=1)
    W = np.mean([np.var(chain, ddof=1) for chain in chains])
    
    var_hat = (1 - 1 / n) * W + B / n
    
    R_hat = np.sqrt(var_hat / W)
    
    return R_hat

def main():

    nwalkers = 5
    nsteps = 100000

    sampler = MCMC(log_prob,proposal,nwalkers)
    
    sampler.run_mcmc(nsteps)
    #print(sampler.getChainParameter(1))
   
    samples = sampler.getChain()
    samples_array = np.array(samples)

    R_hat = convergenceCheck(samples_array)
    print("Gelman-Rubin R-hat:", R_hat)

    all_samples = np.hstack(samples)

    plt.figure(figsize=(10, 6))
    for i, walker_samples in enumerate(samples):
        plt.plot(range(nsteps), walker_samples, alpha=0.6, label=f'Walker {i+1}' if i < 10 else "")
    plt.xlabel("Step")
    plt.xlim(99900,100000)
    plt.ylabel("Position (meters)")
    plt.title("Trajectory of Each Walker Over MCMC Steps")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.show()

    # Optional: Histogram of all samples to approximate the radial probability distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_samples, bins=100, density=True, alpha=0.7, color='b', label="MCMC Samples")
    plt.xlabel("Radius (meters)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Radial Probability Distribution of 3s Orbital (Hydrogen)")
    plt.show()



