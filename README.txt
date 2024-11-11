RadialDist Module README

Thank you for downloading and installing the "RadialDist" package, please make sure that you also have the appropriate "pyproj
ect.toml" file for version 1.0.1 before continuing below. 


The main focus of this package is to perform a Monte Carlo Markov Chain simulation for determining the radial distribution of the 
3s orbital for a ground state hydrogen atom. 

The package can be installed by typing the following in the command line: "pip(3) install fission" which downloads all the corresponding 
modules necessary for the fission module to run properly. In addition please be sure to have pip version 24.2 installed in your virtual 
environment prior to installation of fission.

Following installation, the following scripts can be ran from command line:


main_script

"main_script" will run the MCMC chain for 100000 iterations and displays the evolution of the walker as well as the radial probability distribution as a function of r values.