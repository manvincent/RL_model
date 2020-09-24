#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:38:45 2019

@author: vman
"""
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import time
from defineModel import *
from utilities import *

def unwrap_self(modFit, seedIter, seeds, taskData, numTrials, alpha_prior, beta_prior):
    return modFit.minimizer(seedIter, seeds, taskData, numTrials, alpha_prior, beta_prior)


class Optimizer(object):
    def __init__(self):
        self.xtol = 0.001
        self.ftol = 0.01
        self.maxiter = 1000
        self.disp = False
        return

    def getFit(self, taskData, numTrials):
        # Initialize the model and bounds
        self.initMod = ModelType()
        # Define parameter bounds
        self.boundList = tuple((self.initMod.alpha_bounds, self.initMod.beta_bounds))
        # Define prior distribution
        alpha_prior, beta_prior = self.initMod.paramPriors()
        ### Optimization
        # Set up seed searching
        numSeeds = 100
        seeds = np.zeros([self.initMod.numParams,numSeeds], dtype=float)
        seeds[0,:] = np.random.permutation(np.linspace(self.initMod.alpha_bounds[0], self.initMod.alpha_bounds[1], numSeeds))
        seeds[1,:] = np.random.permutation(np.linspace(self.initMod.beta_bounds[0], self.initMod.beta_bounds[1], numSeeds))
        # Parallelize across seed iterations
        num_cores = multiprocessing.cpu_count()
        parallelResults = Parallel(n_jobs=num_cores)(delayed(unwrap_self)(self, seedIter, seeds, taskData, numTrials, alpha_prior, beta_prior)
                        for seedIter in np.arange(numSeeds))
        return parallelResults

    def minimizer(self, seedIter, seeds, taskData, numTrials, alpha_prior, beta_prior):
        optimResults = minimize(self.posterior,
                                seeds[:,seedIter],
                                args = (taskData, numTrials, alpha_prior, beta_prior),
                                method = 'TNC',
                                bounds = self.boundList,
                                options = dict(disp = self.disp,
                                             maxiter = self.maxiter,
                                             xtol = self.xtol,
                                             ftol = self.ftol))
        return(optimResults)

    def likelihood(self, param, taskData, numTrials):
        # Initialize Q values
        qval = np.ones(2, dtype = float) * np.mean(np.array([0,0.1,0.2,0.3]))
        # Unpack parameter values
        qA, smB = param
        # Initilize trial likelihood
        pChoice = 0
        for tI in np.arange(numTrials):
            if taskData.runReset[tI] == 1: 
                qval = np.ones(2, dtype = float) * np.mean(np.array([0,0.1,0.2,0.3]))
            if ~np.isnan(taskData.respIdx[tI]):
                # Run model simulation
                [_, pOptions] = self.initMod.actor(qval, smB)
                # Get observed response
                respIdx = taskData.respIdx[tI].astype(int)
                # Get likelihood of data | model
                pChoice += np.log(pOptions[respIdx])
                # Observe outcome
                reward = taskData.payOut[tI]
                # Update action value according to the delta rule
                [qval[respIdx],_] = self.initMod.learner(qval[respIdx], qA, reward)                                
            else:
                pChoice += 1e-10
        return pChoice

    def posterior(self, param, taskData, numTrials, alpha_prior, beta_prior):
        # Define likelihood function
        logLike = self.likelihood(param, taskData, numTrials)
        # Add log likelihood and log priors across parameters
        logLike += alpha_prior(param[0])
        logLike += beta_prior(param[1])
        # Compute posterior (neg log) likelihood
        NLL = -1 * logLike
        return NLL

    def transformParams(self):
        self.transfParams = np.zeros(self.numParams,dtype=float)
        # Transform offered values into model parameters
        # Q model parameters
        currBounds = paramBounds()
        self.transfParams[0] = np.max(currBounds[0])/(1+np.exp(-self.transfParams[0])) # Q-learning alpha
        self.transfParams[1] = np.max(currBounds[1])/(1+np.exp(-self.transfParams[1])) # softmax Beta
        return
