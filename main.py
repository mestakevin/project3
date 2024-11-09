import numpy as np
import tqdm
import numpy.random
from matplotlib import pyplot as plt
from custom_mcmc import MCMC

numpy.random.seed(12345)

def posterior(param):

    return 1 / (2 * numpy.pi) ** 0.5 * numpy.exp(     (-1/2 * (param[0] + param[1]) ** 2.0) * numpy.sin(3 * param[2]) ** 2.0)

def proposal(param):
    

    a = numpy.random.uniform(-1,1) + param[0]
    b = numpy.random.uniform(-1,1) + param[1]
    c = numpy.random.uniform(-1,1) + param[2]
    
    return [a,b,c]
        
    
def main():
    sampler = MCMC(posterior,proposal,500)
    initial = [1,1,1]
    
    sampler.run_mcmc(initial,100000)
    #print(sampler.getChainParameter(1))
   
    plt.figure()
    plt.title("Evolution of the walker")
    plt.plot(sampler.getChainParameter(1))
    plt.ylabel('a-value')
    plt.xlabel('Iteration')

    plt.figure()
    plt.title("Posterior samples")
    _ = plt.hist(sampler.getChainParameter(1)[100::100], bins=100)
    plt.show()

    plt.figure()
    plt.title("Evolution of the walker")
    plt.plot(sampler.getChainParameter(2))
    plt.ylabel('b-value')
    plt.xlabel('Iteration')

    
    plt.figure()
    plt.title("Posterior samples")
    _ = plt.hist(sampler.getChainParameter(2)[100::100], bins=100)
    plt.show()

    plt.figure()
    plt.title("Evolution of the walker")
    plt.plot(sampler.getChainParameter(3))
    plt.ylabel('b-value')
    plt.xlabel('Iteration')

    plt.figure()
    plt.title("Posterior samples")
    _ = plt.hist(sampler.getChainParameter(3)[100::100], bins=100)
    plt.show()
main()