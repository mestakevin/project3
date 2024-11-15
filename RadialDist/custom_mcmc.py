import numpy as np
import tqdm
import random


class MCMC:

    def __init__(self, posterior, proposal,walkers):
       
        self.posterior = posterior #posterior function
        self.proposal = proposal #proposal distribution function
        self.walkers = walkers # how many walkers
        self.chain = 0
        self.prob = 0
    
    def getNumWalkers(self):
        return self.walkers
    
    def getPosterior(self):
        return self.posterior
    
    def getProposal(self):
        return self.proposal
    
    def getChain(self):
        return self.chain
    
    def getChainParameter(self,param_num):
        param_chain = []
        for i in self.getChain():
            param_chain.append(i[param_num - 1])
        return param_chain
    
    def run_mcmc(self, iterations):
        a0 = 5.29e-11 
        post_func = self.getPosterior()
        propos_func = self.getProposal()
        num_walkers = self.getNumWalkers()
        #print(num_walkers)
        

        walker_list = [[random.uniform(0, 10 * a0)] for _ in range(num_walkers)]
        samples = [[] for _ in range(num_walkers)]

        for i_num in tqdm.tqdm(range(iterations)):
            for walk_num in range(num_walkers):
                current_position = walker_list[walk_num][-1]
                #print(current_position, type(current_position))
                test_position = propos_func(current_position)

                prob_current = post_func(current_position)
                prob_test = post_func(test_position)
            
                acceptance = np.exp(prob_test - prob_current)
            
                u = np.random.uniform(0,1)
            
                if acceptance >= 1 or u < acceptance:
                    walker_list[walk_num].append(test_position)
                samples[walk_num].append(walker_list[walk_num][-1])
                    

        self.chain = samples
        





        