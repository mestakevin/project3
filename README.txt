RadialDist Module README

Thank you for downloading and installing the "RadialDist" package, please make sure that you also have the appropriate "pyproj
ect.toml" file for version 1.0.1 before continuing below. 


The main focus of this package is to perform a Monte Carlo Markov Chain simulation for determining the radial distribution of the 
3s orbital for a ground state hydrogen atom. 

The package can be installed by typing the following in the command line: "pip(3) install RadialDist" which downloads all the corresponding 
modules necessary for the RadialDist module to run properly. In addition please be sure to have pip version 24.2 installed in your virtual 
environment prior to installation of RadialDist.

Following installation, the following scripts can be ran from command line:


main_script
emcee_script 
autocorr_step 
converge_test 
burn_in 

"main_script" will run the MCMC chain for 100000 iterations and displays the evolution of the walker as well as the radial probability distribution as a function of r values.

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

