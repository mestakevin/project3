import numpy as np
import random
from matplotlib import pyplot as plt
from .custom_mcmc import MCMC
from .emcee_mcmc import run_emcee_param

a0 = 5.29e-11  # Bohr radius in meters


def radial_func(r):
    """
    Supplemental wavefunction of the 3s orbital to be used by the radial_prob_func function

    Parameters:
        r (float): The position at which to evaulate the wavefunction at
    
    Returns:
        (float): Value of the wavefunction evaluated at r
    """
    normalization = 1 / (81 * np.sqrt(3 * (np.pi) * a0**3))
    return normalization * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))


def radial_prob_func(r):
    """
    Radial probabiity density function to be used by log_prob function

    Parameters:
        r (float): The position at which ti evaulate the radial probability density function at
    
    Returns:
        (float): The probability density at the given position
    """
    return (radial_func(r) ** 2 )* r **2

def proposal(r,step_size):
    """
    Proposal function to generate new position of walker to travel given the current position of walker

    Parameters:
        r (float): The current position of a given walker
        step_size (float): Factor by which to scale the size of steps taken by walker when choosing new position to travel to

    Returns:
        new_r (float): New position proposed for the walker to travel to
    """
    new_r = r + step_size * random.gauss(0, a0)
    return new_r

def log_prob(r):
    """
    The log of the posterior probability function, radial_prob_func , that returns the log of the probability at a given position, assuming that the probability at the priors is increasingly small and negligble

    Parameters:
        r (float): The position at which to evaulate function at

    Returns:
        (float): The log of the radial_prob_func evaulated at r 
    """
    if r < 0:
        return -np.inf
    return np.log(radial_prob_func(r) + 1e-100)


def convergenceCheck(chains):
    """
    Computes the Gelman-Rubin convergence statistic for the array of chains produced by the custom MCMC 

    Parameters:
        chains (array): Array containing the array of positions traveled by each walker in the custom MCMC simulation

    Returns:
        R_hat (float): The Gelman-Rubin convergence statistic
    """
    m, n = chains.shape  
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n / (m-1) * (np.sum(np.square(chain_means - overall_mean)))
    W = 0
    for i in range(m):
        s_square = 1 / (n - 1) * np.sum(np.square(chains[i] - chain_means[i]))
        W += s_square 
    W = 1 / m * W
    R_hat = ((1 - 1 / n) * W + B / n) / W
   
    return R_hat

def convergenceCheck_emcee(chains):
    """
    Computes the Gelman-Rubin convergence statistic for the array of chains produced by the emcee package

    Parameters:
        chains (array): Array containing the array of positions traveled by each walker in the emcee package simulation

    Returns:
        R_hat (float): The Gelman-Rubin convergence statistic
    """
    m, n, ndim = chains.shape  
    
    nsteps, nwalkers, ndim = chains.shape
    chain_means = np.mean(chains, axis=0)
    overall_mean = np.mean(chain_means, axis=0)
    B = nsteps * np.sum(np.square(chain_means - overall_mean), axis=0) / (nwalkers - 1)
    W = np.sum(np.var(chains, axis=0, ddof=1), axis=0) / nwalkers
    var_hat = ((nsteps - 1) / nsteps) * W + (B / nsteps)
    R_hat = np.sqrt(var_hat / W)

    return R_hat

def autocorrelation(chain, max_lag=None):
    """
    Autocorrelation function to be used to compute the autocorrelation length of a MCMC simulation

    Parameters:
        chain (array): Array containing the positions traveled by a single walker in the custom MCMC simulation
        max_lag (int): Largest value of the offset of the chain with itself

    Returns:
        (array): Value of autocorrelation function evaluated at each offset up until the max_lag
    """
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)
    if max_lag is None:
        max_lag = n // 2
    
    acf = np.correlate(chain - mean, chain - mean, mode='full') / (var * n)
    return acf[n - 1 : n - 1 + max_lag]

def autocorrelation_length(chain, max_lag=None):
    """
    Computes the autocorrelation length for a single chain of positions of a MCMC simulation

    Parameters:
        chain (array): Array containing the positions traveled by a single walker in the custom MCMC simulation
        max_lag (int): Largest value of the offset of the chain with itself

    Returns:
        (float): The sum of the values of the autocorrelation function which is the autocorrelation length for a single chain
    """
    acf = autocorrelation(chain, max_lag)
    positive_acf = acf[acf > 0] 
    return 1 + 2 * np.sum(positive_acf[1:]) 


def getavgAutoCorrelation(samples_array):
    """
    Computes the average autocorrelation length of a MCMC simulation

    Parameters:
        samples_array (array): Array containing the array of positions traveled by each walker in the custom MCMC simulation

    Returns:
        mean_autocorr_length (float): the average autocorrelation length across all of the walkers in the simulation
    """
    autocorr_lengths = []
    for walker_samples in samples_array:
        acl = autocorrelation_length(walker_samples)
        autocorr_lengths.append(acl)
    mean_autocorr_length = np.mean(autocorr_lengths)
    
    return mean_autocorr_length

def optimal_burnin():
    """
    Generates a plot displaying how the convergence statistic and autocorrelation length vary as the assumed burn-in period varies

    Returns:
        None

    Outputs:
        Displays two plots, one of Convergence Statistic vs Assumed Burn-in Period and another of Average Autocorrelation Length vs Assumed Burn-in Period
    """
    burn_in_list = [0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2]


    nwalkers = 50
    nsteps = 100000
    step_size = 5
    pos_range = [200.0e-10,250e-10]


    sampler = MCMC(log_prob,proposal,nwalkers)
    sampler.run_mcmc(step_size,nsteps,pos_range)

    R_hat_list = []
    autocorr_list = []

    for burnnum in burn_in_list:
        sampler_copy = sampler
        sampler_copy.discard(burnnum)
        burned_samples_array = np.array(sampler_copy.getChain())

        R_hat = convergenceCheck(burned_samples_array)
        R_hat_list.append(R_hat)

        autcorr_length = getavgAutoCorrelation(burned_samples_array)
        autocorr_list.append(autcorr_length)

        all_samples = np.hstack(burned_samples_array)


    print(R_hat_list)
    print(autocorr_list)
    plt.figure(figsize=(10, 6))
    plt.plot(burn_in_list,R_hat_list)
    plt.xlabel("Assumed Burn-in Period ")
    plt.ylabel("Convergence Statistic")
    plt.title("Convergence Statistic vs Assumed Burn-in Period")

    plt.figure(figsize=(10, 6))
    plt.plot(burn_in_list,autocorr_list)
    plt.xlabel("Assumed Burn-in Period ")
    plt.ylabel("Average Autocorrelation Length")
    plt.title("Average Autocorrelation Length vs Assumed Burn-in Period")    
    plt.show()

        


def auto_corr_vs_step_size():
    """
    Generates a plot displaying how the convergence statistic and autocorrelation length vary as the step size factor varies

    Returns:
        None

    Outputs:
        Displays two plots, one of Convergence Statistic vs Step Size and another of Average Autocorrelation Length vs Step Size
    """
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

    #  
        sampler.discard(0.2)
        
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

def emcee_vs_custom():
    """
    Compares the convergence statstic for the custom MCMC simulation and imported emcee package simulation for a given number of walkers, iterations, initial positions, step size, and assumed burn-in period

    Returns:
        None
    Outputs:
        Prints the convergence statistic of the emcee and custom MCMC simulation
    """
    nwalkers = 50
    nsteps = 100000
    step_size = 5
    pos_range = [0.0e-10,10.0e-10]

    emcee_samples = run_emcee_param(nwalkers, nsteps,pos_range[0],pos_range[1])
    

    R_hat_emcee = convergenceCheck_emcee(emcee_samples)

    sampler = MCMC(log_prob,proposal,nwalkers)
    
    sampler.run_mcmc(step_size,nsteps,pos_range)
    sampler.discard(0.2)
    custom_samples = sampler.getChain()

    cutsom_samples_array = np.array(custom_samples)

    R_hat_custom = convergenceCheck(cutsom_samples_array)


    print("R_hat custom:",R_hat_custom,"\t R_hat emcee:",R_hat_emcee)



def num_input(prompt):
    """
    Obtains a number from the user to be used for simulating

    Parameters:
        prompt (str): Message to be displayed to prompt user to input a number
    Returns:
        float: User inputted number
    """
    try:
        num = float(input(prompt))
    except ValueError:
        num = num_input("Invalid input, please enter a number: ")
    return num

def main_program():
    """
    Performs the custom MCMC simulation given user inputted parameters

    Returns:
        None

    Outputs:
        Displays the convergence statistic and the average autocorrelation length of the MCMC simulation as well as generating two plots one of the Trajectory of Each Walker Over MCMC Steps and another of the Radial Probability Distribution of 3s Orbital 
    
    """

    nwalkers = int(num_input("How many walkers would you like to simulate?\n>"))
    nsteps = int(num_input("How many iterations would you like to simulate for?\n>"))
    step_size = num_input("What value for 'step_size' would you like to use?\n>")
    #pos_range = [0.0e-10,1.5e-10]
    #pos_range = [200.0e-10,250e-10]
    pos_range= [num_input("What would you like to set the lower bound of inital positions to?\n>"),num_input("What would you like to set the upper bound of inital positions to?\n>")]
    burn_in = num_input("What would you like to set the burn-in period to?\n>")
    
    sampler = MCMC(log_prob,proposal,nwalkers)
    sampler.run_mcmc(step_size,nsteps,pos_range)
    sampler.discard(burn_in)
    samples = sampler.getChain()
    all_samples_array = np.array(samples)


    R_hat = convergenceCheck(all_samples_array)
    print("Gelman-Rubin R-hat:", R_hat)
    autcorr_length = getavgAutoCorrelation(all_samples_array)
    print("The average autocorrelation length for",nwalkers,"walkers,",nsteps,"iterations, with a step size of",step_size,"is: ",autcorr_length)
   
    all_samples = np.hstack(samples)

    plt.figure(figsize=(10, 6))
    for i, walker_samples in enumerate(samples):
        plt.plot(range(int(nsteps*(1-burn_in))), walker_samples, alpha=0.6, label=f'Walker {i+1}' if i < 10 else "")
    plt.xlabel("Step")
    plt.xlim(0,1000)
    plt.ylabel("Position (meters)")
    plt.title("Trajectory of Each Walker Over MCMC Steps")
    plt.legend(loc="upper right", ncol=2, fontsize=8)

    plt.figure(figsize=(8, 5))
    plt.hist(all_samples, bins=100, density=True, alpha=0.7, color='b', label="MCMC Samples")
    plt.xlabel("Radius (meters)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Radial Probability Distribution of 3s Orbital (Hydrogen)")
    plt.show()
