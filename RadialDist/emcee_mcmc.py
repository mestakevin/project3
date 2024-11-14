import numpy as np
import emcee
import matplotlib.pyplot as plt

# Constants
np.random.seed(12345)
a0 = 5.29e-11  # Bohr radius in meters

##----------------------------------------------##
def radial_func(r):
    normalization = 1 / (81 * np.sqrt(3 * np.pi * a0**3))
    return (normalization * (27 - (18 * (r / a0)) + (2 * (r / a0)**2))) * np.exp(-r / (3 * a0))
##----------------------------------------------##
def radial_prob_func(r):
    # Make sure r is treated as a scalar
    r = r[0] if isinstance(r, list) else r
    value = ((radial_func(r) ** 2) * r ** 2)
    return value
##----------------------------------------------##
def log_prob(r):
    if r[0] < 0:
        return -np.inf
    return np.log(radial_prob_func(r[0]) + 1e-100)
##----------------------------------------------##
def main():
    # Set up MCMC parameters
    ndim = 1
    nwalkers = 50
    nsteps = 10000

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
    counts, bins, _ = plt.hist(samples, bins=100, density=True, alpha=0.7, color='b', label="MCMC Samples")

    # Calculate bin centers and scale by bin width for better comparison with the PDF
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_width = bins[1] - bins[0]

    # Overlay theoretical distribution
    r_values = np.linspace(0, 30 * a0, 1000)
    theoretical_distribution = np.array([radial_prob_func(r) for r in r_values])

    # Ensure that `bin_width` is defined explicitly as a float
    bin_width = float(bin_width)

    # Normalize the theoretical distribution
    theoretical_distribution /= np.sum(theoretical_distribution * bin_width)

    plt.plot(r_values, theoretical_distribution, 'r-', label="Theoretical 3s radial distribution")
    plt.xlabel("Radial distance r (m)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.show()




    # Autocorrelation time and effective sample size
    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time:", tau)
        print("Effective sample size:", len(samples) / tau)
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time could not be estimated reliably.")
##----------------------------------------------##
main()
