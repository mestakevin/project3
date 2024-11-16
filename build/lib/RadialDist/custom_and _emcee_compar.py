import numpy as np
import matplotlib.pyplot as plt
from custom_mcmc import MCMC  # Import your custom MCMC class
import emcee

# Constants
a0 = 5.29e-11  # Bohr radius in meters
##----------------------------------------------##
def radial_func(r):
    normalization = 1 / (81 * np.sqrt(3 * np.pi * a0**3))
    return (normalization * (27 - (18 * (r / a0)) + (2 * (r / a0)**2))) * np.exp(-r / (3 * a0))
##----------------------------------------------##
def radial_prob_func(r):
    r = r[0] if isinstance(r, list) else r
    return (radial_func(r) ** 2) * r ** 2
##----------------------------------------------##
def proposal(param):
    new_r = param[0] + np.random.normal(0, a0 * 3)
    return [new_r]
##----------------------------------------------##
def log_prob(r):
    if r[0] < 0:
        return -np.inf
    return np.log(radial_prob_func(r[0]) + 1e-100)
##----------------------------------------------##
def run_custom_mcmc():
    custom_sampler = MCMC(radial_prob_func, proposal, 1)  # Single walker for simplicity
    initial = [5 * a0]
    custom_sampler.run_mcmc(initial, 10000)
    return custom_sampler.getChainParameter(1)
##----------------------------------------------##
def run_emcee_mcmc():
    ndim = 1
    nwalkers = 50
    nsteps = 10000
    initial_positions = np.array([5 * a0 + 1e-10 * np.random.randn(ndim) for i in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_positions, nsteps, progress=True)
    samples = sampler.get_chain(discard=int(nsteps * 0.2), flat=True)
    return samples.flatten()  # Flatten the array to 1D for easy plotting
##----------------------------------------------##
def main():
    # Run custom MCMC and emcee MCMC
    custom_samples = run_custom_mcmc()
    emcee_samples = run_emcee_mcmc()

    # Plot histograms
    plt.figure()
    plt.hist(custom_samples, bins=100, density=True, alpha=0.7, label="Custom MCMC", color='b')
    plt.hist(emcee_samples, bins=100, density=True, alpha=0.5, label="emcee MCMC", color='orange')

    # Overlay theoretical distribution for reference
    r_values = np.linspace(0, 30 * a0, 1000)
    theoretical_distribution = np.array([radial_prob_func(r) for r in r_values])
    bin_width = (r_values[1] - r_values[0])
    theoretical_distribution /= np.sum(theoretical_distribution * bin_width)
    plt.plot(r_values, theoretical_distribution, 'r-', label="Theoretical 3s radial distribution")

    # Labels and legend
    plt.xlabel("Radial distance r (m)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.title("Comparison of Custom MCMC and emcee MCMC")
    plt.show()
##----------------------------------------------##
main()
