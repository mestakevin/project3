import numpy as np
import tqdm


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
    
    def run_mcmc(self, prior, iterations):

        post_func = self.getPosterior()
        propos_func = self.getProposal()
        

        chain = [prior]
        prob = [post_func(prior)]
    
        for i in tqdm.tqdm(range(iterations)):
            param_test = propos_func(chain[-1])
            prob_test = post_func(param_test)
            
            acceptance = prob_test / prob[-1]
            u = np.random.uniform(0,1)
            
            if u <= acceptance:
                chain.append(param_test)
                prob.append(prob_test)

        self.chain = chain
        self.prob = prob





        