import numpy as np
import tqdm
import numpy.random
import random
from matplotlib import pyplot as plt
from .custom_mcmc import MCMC

#numpy.random.seed(12345)
a0 = 5.29e-11  # Bohr radius in meters


def radial_func(r):
    #r = param[0]
    normalization = 1 / (81 * np.sqrt(3 * (np.pi) * a0**3))
    return normalization * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))


def radial_prob_func(r):
    #r = param[0]
    return (radial_func(r) ** 2 )* r **2

def proposal(r,step_size):

    new_r = r + step_size * random.gauss(0, a0)
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

def autocorrelation(chain, max_lag=None):
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)
    if max_lag is None:
        max_lag = n // 2
    
    acf = np.correlate(chain - mean, chain - mean, mode='full') / (var * n)
    return acf[n - 1 : n - 1 + max_lag]

def autocorrelation_length(chain, max_lag=None):
    acf = autocorrelation(chain, max_lag)
    positive_acf = acf[acf > 0] 
    return 1 + 2 * np.sum(positive_acf[1:]) 


def getavgAutoCorrelation(samples_array):
    autocorr_lengths = []
    for walker_samples in samples_array:
        acl = autocorrelation_length(walker_samples)
        autocorr_lengths.append(acl)
    mean_autocorr_length = np.mean(autocorr_lengths)
    
    return mean_autocorr_length


def auto_corr_vs_step_size():
    step_size_range = [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0]
    avgautocorr_list = []
    rhat_list =[]
    for i in step_size_range:
        nwalkers = 50
        nsteps = 100000
        step_size = i
        pos_range = [0.0e-10,1.5e-10]
        sampler = MCMC(log_prob,proposal,nwalkers)
    
        sampler.run_mcmc(step_size,nsteps,pos_range)
    #   print(sampler.getChainParameter(1))
   
        samples = sampler.getChain()
        samples_array = np.array(samples)

        R_hat = convergenceCheck(samples_array)
        rhat_list.append(R_hat)
        avgautocorr_list.append(getavgAutoCorrelation(samples_array))

    plt.figure(figsize=(10, 6))
    plt.plot(step_size_range,avgautocorr_list)
    plt.xlabel("Step Size")
    plt.ylabel("Average Autocorrelation Length")
    plt.title("Average Autocorrelation Length vs Step Size")

    plt.figure(figsize=(10, 6))
    plt.plot(step_size_range,rhat_list)
    plt.xlabel("Step Size")
    plt.ylabel("Convergence Statistic")
    plt.title("Convergence Statistic vs Step Size")    
    plt.show()

def main():

    nwalkers = 5
    nsteps = 100000
    step_size = 2
    pos_range = [0.0e-10,1.5e-10]
    sampler = MCMC(log_prob,proposal,nwalkers)
    
    sampler.run_mcmc(step_size,nsteps,pos_range)
    #print(sampler.getChainParameter(1))
   
    samples = sampler.getChain()
    samples_array = np.array(samples)

    R_hat = convergenceCheck(samples_array)
    print("Gelman-Rubin R-hat:", R_hat)


    getavgAutoCorrelation(samples_array)

    all_samples = np.hstack(samples)

    plt.figure(figsize=(10, 6))
    for i, walker_samples in enumerate(samples):
        plt.plot(range(nsteps), walker_samples, alpha=0.6, label=f'Walker {i+1}' if i < 10 else "")
    plt.xlabel("Step")
    #plt.xlim(0,1000)
    plt.ylabel("Position (meters)")
    plt.title("Trajectory of Each Walker Over MCMC Steps")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.show()

    
    plt.figure(figsize=(8, 5))
    plt.hist(all_samples, bins=100, density=True, alpha=0.7, color='b', label="MCMC Samples")
    plt.xlabel("Radius (meters)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Radial Probability Distribution of 3s Orbital (Hydrogen)")
    plt.show()



