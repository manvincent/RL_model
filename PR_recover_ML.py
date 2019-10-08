#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:56:26 2019

@author: vman
"""

## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
# Numpy options
np.seterr(over='warn')
np.set_printoptions(threshold=sys.maxsize)
import pickle
import os
os.chdir("C:\\Users\\Brenden\\OneDrive - California Institute of Technology\\PhD\\O'Doherty Lab\\RL_model")
from optimizer import *
from utilities import *


### Recovery script
def initRecover(numTrials):
    modelName = 'base_delta'
    homeDir = "C:\\Users\\Brenden\\OneDrive - California Institute of Technology\\PhD\\O'Doherty Lab\\RL_model\\param_recov"
    datDir =  homeDir + os.sep + 'Generate' + os.sep + 'Trial_'+str(numTrials)
    outDir = homeDir + os.sep + 'Recover'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    numParams = 2
    initDict = dict2class(dict(modelName = modelName,
                           homeDir = homeDir,
                           datDir = datDir,
                           outDir = outDir))
    return(initDict)


def fitModel():    
    # Loop through different 'experiments' of varying trial numbers
    recovCorrDF = pd.DataFrame() 
    recovSamples = np.empty(len(np.arange(1,16)), dtype=object)
    for sampleIdx, numTrials in enumerate(np.arange(1,16)*10):
        # Print current sample size 
        print(f'Sample size: {numTrials}')
        # Initialize the model fitting object
        initDict = initRecover(numTrials)
        # Create structures for storing estimates across iterations (subjects)
        numIter = 100
        sampleDF = pd.DataFrame()
        # Loop through iterations
        for iterID in np.arange(numIter):
            # Start the timer
            tic = time.time()
            # Load procedure
            inFile = 'sim_' + str(iterID+1)
            # Load in data
            dataFile = load_obj(initDict.datDir + os.sep + inFile + '.pkl')
            genModel = dataFile['Model']
            genTask = dataFile['Task']
            # Unpack the task sessions
            taskData = unpackTask(genTask)
            # Print current fit details
            print(f'Iteration No: {iterID}')
             # Initialize the optimizer
            initOptimizer = Optimizer()
            # Run the optimizer
            parallelResults = initOptimizer.getFit(taskData, numTrials)
            # Store estimates from this seed iteration
            fitParams = np.array([parallelResults[s]['x'] for s in np.arange(len(parallelResults))])
            fitNLL = np.array([parallelResults[s]['fun'] for s in np.arange(len(parallelResults))])
            # Store best fitted parameter likelihoods and values
            minIdx = np.argmin(fitNLL)
            recovParams = fitParamContain(fitNLL[minIdx]).instr_deltaLearner_fitParams(fitParams[minIdx])
            # Store generative parameter values
            genParams = genModel.genParams
            # Append to dataframe
            sampleDF = sampleDF.append({'gen_alpha': genParams.alpha_i,
                                        'recov_alpha': recovParams.alpha_i,
                                        'gen_beta': genParams.beta_i,
                                        'recov_beta': recovParams.beta_i}, ignore_index=True)
        # Save the gen/recov parameters for this 'experiment'    
        sampleDF.to_csv(initDict.outDir + os.sep + 'n_' + str(numTrials) + '_paramRecov.csv',index=False)
        # Compute the correlations between the generated and recovered parameters 
        alpha_corr = corr(sampleDF.gen_alpha, sampleDF.recov_alpha)[0]
        beta_corr = corr(sampleDF.gen_beta, sampleDF.recov_beta)[0]
        recovCorrDF = recovCorrDF.append({'sample_n': numTrials,
                                          'alpha': alpha_corr,
                                          'beta': beta_corr}, ignore_index=True)
    # Save parameter recover results
    recovCorrDF.to_csv(initDict.homeDir + os.sep + 'paramRecov_results.csv', index=False)    
    return 


def unpackTask(genTask):
    # Get choice attributes
    highChosen = np.hstack([genTask.sessionInfo[i].highChosen for i in np.arange(genTask.numSessions)])
    selectedStim = np.hstack([genTask.sessionInfo[i].selectedStim for i in np.arange(genTask.numSessions)])
    respIdx = np.hstack([genTask.sessionInfo[i].sessionResponses.respKey for i in np.arange(genTask.numSessions)])
    # Get reversal attributes
    reverseStatus = np.hstack([genTask.sessionInfo[i].reverseStatus for i in np.arange(genTask.numSessions)])
    reverseTrial = np.hstack([genTask.sessionInfo[i].reverseTrial for i in np.arange(genTask.numSessions)])
    # Get stimulus attributes
    pWin = np.hstack([genTask.sessionInfo[i].stimAttrib.pWin for i in np.arange(genTask.numSessions)])
    isHigh = np.hstack([genTask.sessionInfo[i].stimAttrib.isHigh for i in np.arange(genTask.numSessions)])
    isSelected = np.hstack([genTask.sessionInfo[i].stimAttrib.isSelected for i in np.arange(genTask.numSessions)])
    isWin = np.hstack([genTask.sessionInfo[i].stimAttrib.isWin for i in np.arange(genTask.numSessions)])

    # Get outcome attributes
    payOut = np.hstack([genTask.sessionInfo[i].payOut for i in np.arange(genTask.numSessions)])
    return dict2class(dict(highChosen = highChosen,
                           selectedStim = selectedStim,
                           respIdx = respIdx,
                           reverseStatus = reverseStatus,
                           reverseTrial = reverseTrial,
                           pWin = pWin,
                           isHigh = isHigh,
                           isSelected = isSelected,
                           isWin = isWin,
                           payOut = payOut))
    
if __name__ == "__main__":
    fitModel()
