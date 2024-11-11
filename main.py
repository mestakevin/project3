import numpy as np
import tqdm
import numpy.random
import random
from matplotlib import pyplot as plt
from custom_mcmc import MCMC

numpy.random.seed(12345)
a0 = 5.29e-11  # Bohr radius in meters


def radial_func(param):
    r = param[0]
    normalization = 1 / (81 * np.sqrt(3 * a0**3))
    return normalization * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))


def radial_prob_func(param):
    r = param[0]
    return (radial_func(param) ** 2 )* r **2

def proposal(param):
    new_r = param[0] + random.gauss(0, a0)
    return [new_r]
        
    
def main():
    sampler = MCMC(radial_prob_func,proposal,500)

    initial = [5 * a0]
    
    sampler.run_mcmc(initial,100000)
    #print(sampler.getChainParameter(1))
   
    plt.figure()
    plt.title("Evolution of the walker")
    plt.plot(sampler.getChainParameter(1))
    plt.ylabel('r-value')
    plt.xlabel('Iteration')

    plt.figure()
    plt.title("Posterior samples")
    _ = plt.hist(sampler.getChainParameter(1)[100::100], bins=100)
    plt.show()

main()


