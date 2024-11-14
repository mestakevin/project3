import numpy as np
import emcee
import matplotlib.pyplot as plt

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
def main():
    # Set up MCMC parameters
    a0 = 5.29e-11
    ndim = 1
    nwalkers = 50
    nsteps = 5000
    
    initial_positions = np.array([5 * a0 + 1e-10 * np.random.randn(ndim) for i in range(nwalkers)])
    
    # Run emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_positions, nsteps, progress=True)
    samples = sampler.get_chain(discard=int(nsteps * 0.2), flat=True)
    
    # Plot the trace of the first walker
    plt.figure()
    plt.title("Trace of the first walker")
    plt.plot(sampler.get_chain()[:, 0, 0])
    plt.xlabel("Iteration")
    plt.ylabel("Radial distance r (m)")
    plt.show()
    
    # Plot histogram of the posterior samples
    plt.figure()
    plt.title("Posterior samples of radial distance")
    plt.hist(samples, bins=100, density=True, alpha=0.7, color='b')
    plt.xlabel("Radial distance r (m)")
    plt.ylabel("Probability density")
    
    # Overlay theoretical distribution
    r_values = np.linspace(0, 20 * a0, 1000)
    theoretical_distribution = [radial_prob_func(r) for r in r_values]
    plt.plot(r_values, theoretical_distribution, 'r-', label="Theoretical 3s radial distribution")
    plt.legend()
    plt.show()

##-----------------------------------------------------------##
main()
