import numpy as np
import tqdm
import random


class MCMC:

    def __init__(self, posterior, proposal,walkers):
        """
        A class representing the custom Markov chain Monte Carlo (MCMC) simulation using the Metropolis Hastings algorithm 

        Attributes:
            posterior (function): The posterior distribution function used for the MCMC simulation
            proposal (function): The proposal function to determine new steps within the parameter space
            walkers (int): The number of walkers to simulate stepping through the parameter space
            chain (array): An array containing the list of accepted positions for each walker created
        """
        self.posterior = posterior #posterior function
        self.proposal = proposal #proposal distribution function
        self.walkers = int(walkers) # how many walkers
        self.chain = 0

    
    def getNumWalkers(self):
        """
        Obtains the number of number of walkers in the MCMC simulation

        Returns:
            int: Number of walkers 
        """
        return self.walkers
    
    def getPosterior(self):
        """
        Obtains the posterior function used in the MCMC simulation

        Returns:
            function: The posterior function used in MCMC simulation
        """
        return self.posterior
    
    def getProposal(self):
        """
        Obtains the proposal function used in the MCMC simulation

        Returns:
            function: The proposal function used in MCMC simulation
        """
        return self.proposal
    
    def getChain(self):
        """
        Obtains the chain of the simulated MCMC 

        Returns:
            array: The chain of the simulated MCMC
        """
        return self.chain
    
    def run_mcmc(self, step_size,iterations,pos_range):
        """
        Performs the MCMC given the step size, number of iterations, and range of inital positions

        Parameters:
            step_size (float): Factor by which to scale the size of steps taken by walkers when choosing new positions to travel to
            iterations (int): The number of iterations to run the MCMC simulation for
            pos_range (list): 2-item list containing the lower and upper bound of possible positions to initalize walkers between

        Returns:
            None

        Outputs:
            MCMC object's chain attribute is updated to include all of the positions that all the walkers traveled to
        """
        a0 = 5.29e-11 
        post_func = self.getPosterior()
        propos_func = self.getProposal()
        num_walkers = self.getNumWalkers()
        lower = pos_range[0]
        upper = pos_range[1]
        #print(num_walkers)
        
        walker_list = [[random.uniform(lower, upper)] for _ in range(int(num_walkers))]
        #walker_list = [[random.uniform(0, 10 * a0)] for _ in range(num_walkers)]
        samples = [[] for _ in range(num_walkers)]

        for i_num in tqdm.tqdm(range(iterations)):
            for walk_num in range(num_walkers):
                current_position = walker_list[walk_num][-1]
                #print(current_position, type(current_position))
                test_position = propos_func(current_position,step_size)

                prob_current = post_func(current_position)
                prob_test = post_func(test_position)
            
                acceptance = np.exp(prob_test - prob_current)
            
                u = np.random.uniform(0,1)
            
                if acceptance >= 1 or u < acceptance:
                    walker_list[walk_num].append(test_position)
                samples[walk_num].append(walker_list[walk_num][-1])
                    

        self.chain = samples
        

    def discard(self,percent):
        """
        Dicards a fraction of initial positions for all walkers

        Parameters:
            percent (float): Decimal that can start at 0 but cannot be equal to or greater than 1

        Returns:
            None

        Outputs:
            MCMC object's chain attribute is updated to only include positions of walkers after the fraction of discarded positions
        """
        burn_in_length = int(len(self.getChain()[0]) * percent)

        self.chain = [walker[burn_in_length: ] for walker in self.getChain()]

        





        