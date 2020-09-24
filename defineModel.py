### Instrumental model
	# Q(a) = Q(a) + alpha * rPE
    # rPE = R(expect) - R(observed)
	# P(a) = 1 / ( 1 + exp(beta * diff(Q(s)))
## Pavlovian model
    # V(s) = V(s) + alpha + rPE
    # P(c) = exp(beta * V(si)) / [ exp(beta * V(sj)) + exp(beta * V(sk)) ]
    # P(c) =  np.exp(p) / np.sum(np.exp(p))
# Import modules
import numpy as np
from itertools import compress
from scipy.stats import gamma, beta, uniform
# Define model types
class ModelType(object):

    def __init__(self):
        self.numParams = 2
        # Define parameter lower and upper bounds
        self.alpha_bounds = (0.01, 0.99)
        self.beta_bounds = (1, 20)
        self.alpha_a = self.alpha_b = 1.4
        self.beta_shape = 4.83
        self.beta_scale = 0.73
        return

    def genPriors(self, arr):
        # Distributions
        alpha_genDistr = beta(self.alpha_a, self.alpha_b)
        beta_genDistr = gamma(self.beta_shape, self.beta_scale)
        # Ranges
        alpha_range = alpha_genDistr.cdf(self.alpha_bounds)
        beta_range = beta_genDistr.cdf(self.beta_bounds)
        alpha_genVal = alpha_genDistr.ppf(np.linspace(*alpha_range, num=arr))
        beta_genVal = beta_genDistr.ppf(np.linspace(*beta_range, num=arr))
        # Prior distribution for learning rate
        #alpha_genDistr = np.round(np.random.beta(self.alpha_a, self.alpha_b, arr),3)
        # Prior distribution of softmax beta
        #beta_genDistr = np.round(np.random.gamma(self.beta_shape, self.beta_scale, arr),3)
        return alpha_genVal, beta_genVal

    def paramPriors(self):
        # Prior distribution for learning rate
        alpha_logpdf = lambda x: np.sum(np.log(beta.pdf(x, self.alpha_a, self.alpha_b)))
        alpha_logpdf.__name__ = "qA_logpdf"
        # Prior distribution of softmax beta
        beta_logpdf = lambda x: np.sum(np.log(gamma.pdf(x, self.beta_shape, loc=0, scale=self.beta_scale)))
        beta_logpdf.__name__ = "smB_logpdf"
        return alpha_logpdf, beta_logpdf

    def learner(self, Q, alpha, reward):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            Q: Model learnt Q value of chosen alternative
            alpha: delta-rule learning rate; scalar
            reward: observed reward outcome 
        """
        RPE = (reward - Q)
        Q =  Q + alpha * RPE
        return(Q, RPE)

    def actor(self, Q, beta):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            Q: the expected action value, computed by learner (for all choices; vector)
            beta: softmax inverse temperature; scalar
        """
        # Specify number of parameters
        reward = int()
        RPE = int()
        # Action selection through logistic function
        pOpt1 = 1 / float( 1 + np.exp(beta *(Q[1] - Q[0])))
        pOptions = np.array([pOpt1,1-pOpt1])
        ## Note: if left response is assigned to index 0 and right to index 1 in [selectedIndex]
        ## (for taskStruct['Q']), then this returns p(right choice)
        ## Pick an option given the softmax probabilities
        respIdx = np.where(np.cumsum(pOptions) >= np.random.rand(1))[0][0]
        # output: 0 means left choice, 1 means right choice
        return(respIdx, pOptions)


    def returnGen(self):
        # Pack up and return general parameters
        genParams = np.empty(2, dtype=object)
        genParams[0] = self.alpha
        genParams[1] = self.beta
        return genParams


# Set up a container for model parameter estimates (fits)
class fitParamContain():
    def __init__(self, fitlikelihood):
        self.fitlikelihood = fitlikelihood
        return

    def instr_deltaLearner_fitParams(self, fitParams):
        self.alpha_i = fitParams[0]
        self.beta_i = fitParams[1]
        return self


# Set up a container for genereated parameters
class genParamContain():
    def __init__(self):
        return

    def instr_deltaLearner_genParams(self, currParams):
        self.alpha_i = currParams[0]
        self.beta_i = currParams[1]
        return self
