RadialDist Module README

Thank you for downloading and installing the "RadialDist" package, please make sure that you also have the appropriate 
"pyproject.toml" file for version 2.1.3 before continuing below. 


The main focus of this package is to perform a Monte Carlo Markov Chain simulation for determining the radial distribution 
of the 3s orbital for a ground state hydrogen atom. This package also compares the performance of a custom MCMC simulation 
to the already existing emcee package in modeling the radial distribution desired. 

The package can be installed by typing the following in the command line: "pip(3) install RadialDist" which downloads all 
the corresponding modules necessary for the RadialDist module to run properly. In addition please be sure to have pip 
version 24.2 installed in your virtual environment prior to installation of RadialDist.

Following installation, the following scripts can be ran from command line:

main_script
emcee_script 
autocorr_step 
converge_test 
burn_in 

"main_script" will run the custom MCMC chain according to parameters inputted by the user. A sample input is shown below:

############################################################################
How many walkers would you like to simulate?
>50
How many iterations would you like to simulate for?
>10000
What value for 'step_size' would you like to use?
>5
What would you like to set the lower bound of inital positions to?
>200e-10
What would you like to set the upper bound of inital positions to?
>250e-10
What would you like to set the burn-in period to?
>0
############################################################################

100%|███████████████████████████████████| 10000/10000 [00:03<00:00, 3042.16it/s]
Gelman-Rubin R-hat: 1.415337820546066
The average autocorrelation length for 50 walkers, 10000 iterations, with a step size of 5.0 is:  311.9732675981156

############################################################################

The values for the number of walkers and iterations must be positive integers. Increasing the number of walkers for the
simulation will improve the accuracy of the simulation but will also increase computation time. Increasing the number of iterations will 
also improve the accuracy of the simulation but also increases the computation time. It is recommended to perform the simulation 
with at least 20 walkers and for at least 50,000 iterations.

The value of step size must be a positive float and can be used to tune the exploration of the space of the walkers. A value less than 1
means that the walkers will need significantly more iterations to explore the entire space while larger values allow for quicker exploration of 
the space but should be less than 20 so as to not miss the ideal space for the desired distribution.

The values for the lower and upper bound must be positive floats but can include 0. These two numbers control between what positions
should the walkers' positions be initialized at according to a random uniform distribution. It is recommended to initialize the walkers between 
0 and 15.0e-10 as this where the radial probability distribution is located at. However values in the region of 200e-10 to 300e-10 can also be used 
but will require some time for the chains to be burned in sufficiently. This also requires a step size of around 5 and 100,000 iterations to 
eventually reach a point where the chains have converged.

Lastly, the value of the burn-in period must be a positive float including 0 but no greater than 1. This represents what fraction of initial positions
to discard when plotting the positions of the walkers and calculating convergence and auto-correlation length of the simulation. A burn-in period of
0.2 is enough for most simulations to discard non-ideal positions but increasing it past 0.5 may cause in poor results.

Upon completion of the simulation the Gelman-Rubin convergence statistic and average auto-correlation length are displayed
in the terminal and two plots of the walkers' positions and the resulting calculated probability distribution are generated.

"emcee_script" will run the emcee version of the MCMC simulation using 50 walkers, 100,000 iterations, and a burn-in period of 20%.
The walkers are initialized at positions according to 1e2 * a0 + 1e-10 * np.random.randn(1) and thus start sufficiently far from the target distribution. After
the simulation is completed the auto-correlation length is displayed to the user and two plots are generated, a trace plot of the first walker and another plot
of the radial probability distribution obtained after the simulation with the analytical function overlay-ed for comparison. 

"autocorr_step" will run the custom MCMC simulation using 50 walkers, 100,000 iterations, a burn-in period of 20% and walkers initialized at positions between
0.0e-10 and 1.5e-10. This script will perform the MCMC simulation for varying step sizes (1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0)
and create two plots displaying how the average auto-correlation length and the convergence statistic vary as a function of step size. 

"converge_test" will perform both the custom MCMC and emcee MCMC simulations using 50 walkers, 100,000 iterations, a burn-in period of 20%, and step size 
of 5. The range of positions that both methods initialized the walkers between is 0.0e-10 and 10.0e-10. Upon completing both simulations the convergence
statistic is displayed in the terminal for comparison. 

"burn_in" will run the custom MCMC simulation using 50 walkers, 100,000 iterations, a step size of 5 and walkers initialized at positions between
200.0e-10 and 250.0e-10. This script will perform the MCMC simulation for varying assumed burn-in periods (0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.20)
and create two plots displaying how the average auto-correlation length and the convergence statistic vary as a function of assumed burn-in period. 

