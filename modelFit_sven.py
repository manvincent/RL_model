#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:28:14 2020

@author: vman
"""



## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import pandas as pd
from scipy.stats import pearsonr as corr
import sys
# Numpy options
np.seterr(over='warn')
np.set_printoptions(threshold=sys.maxsize)
import pickle
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/scripts')
from onsets import *
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling/rl')
from optimizer import *
from utilities import *

### Recovery script
def initRecover():
    modelName = 'rl'
    homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis'
    datDir =  f'{homeDir}/data_sven'
    outDir = f'{homeDir}/modelling/{modelName}/model_results'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    initDict = dict2class(dict(modelName = modelName,
                           homeDir = homeDir,
                           datDir = datDir,
                           outDir = outDir))
    return(initDict)


def fitModel():
    # Specify experiment info
    #subList = np.array([2,3,4,5])
    subList = np.array([10,20])
    sessID = np.array([[12,13,16,18,19,31],
                      [21,22,25,26,27,28]])
    # Initialize the model fitting object
    initDict = initRecover()
    # Create structures for storing estimates across subjects
    sampleDF = pd.DataFrame()
    for subIdx, subID in enumerate(subList):
        # Load in data
        inFile = f'subs{subID}_behav.csv'
        subDF = pd.read_csv(f'{initDict.datDir}/{inFile}')
        for day in sessID[subIdx]:
            dayDF = subDF[subDF.sessID == day]
            # Unpack the task sessions
            taskData = unpackTask(dayDF)
            numTrials = taskData.numTrials                
             # Initialize the optimizer
            initOptimizer = Optimizer()
            # Run the optimizer
            parallelResults = initOptimizer.getFit(taskData, numTrials)
            # Store estimates from this seed iteration
            fitParams = np.array([parallelResults[s]['x'] for s in np.arange(len(parallelResults))])
            fitNLL = np.array([parallelResults[s]['fun'] for s in np.arange(len(parallelResults))])
            # Store best fitted parameter likelihoods and values
            minIdx = np.argmin(fitNLL)
            fitParams = fitParamContain(fitNLL[minIdx]).instr_deltaLearner_fitParams(fitParams[minIdx])
            print(f'subID: {subID}, day: {day}, LR: {fitParams.alpha_i}, smB: {fitParams.beta_i}')
            # Append fitted parameters to dataframe
            sampleDF = sampleDF.append({'subID': int(subID),
                                        'day': int(day),
                                        'fit_alpha': fitParams.alpha_i,
                                        'fit_beta': fitParams.beta_i}, ignore_index=True)
    
            # Simulate using retrieved parameter to compute computational variables
            RPE = np.zeros(numTrials, dtype = float)
            Q = np.empty((2,numTrials), dtype = float)
            chosenQ = np.empty(numTrials, dtype = float)
            unchosenQ = np.empty(numTrials, dtype = float)
            simResp = np.empty(numTrials, dtype = float)
            # Initialize qval
            qval = np.ones(2, dtype = float) * 0.5
            for tI in np.arange(numTrials):
                if taskData.runReset[tI] == 1: 
                    qval = np.ones(2, dtype = float) * 0.5
                # Retreive observed (empirical) choice
                respIdx = taskData.respIdx[tI]
                unchosenIdx = 1-taskData.respIdx[tI]
                # Have model simulate response given parameters
                [simResp[tI], _] = ModelType().actor(qval, fitParams.beta_i)
                if ~np.isnan(respIdx):
                    # Store computational variables
                    Q[:,tI] = qval
                    chosenQ[tI] = qval[int(respIdx)]
                    unchosenQ[tI] = qval[int(unchosenIdx)]
                    # Retrieve observed reward
                    reward = taskData.reward[tI]
                    # Update learner
                    [qval[int(respIdx)], RPE[tI]] = ModelType().learner(qval[int(respIdx)], fitParams.alpha_i, reward)
                else:
                    Q[:,tI] = Q[:,tI-1]
                    chosenQ[tI] = np.nan
                    unchosenQ[tI] = np.nan
            # Create dataframe of model-based variables
            compVarDF = pd.DataFrame({'qA': fitParams.alpha_i,
                                      'smB': fitParams.beta_i,
                                      'RPE': RPE,
                                      'absRPE': np.abs(RPE),
                                      'stim1_Q': Q[0],
                                      'stim2_Q': Q[1],
                                      'chosenQ': chosenQ,
                                      'unchosenQ': unchosenQ,
                                      'simResp': simResp})
            compVarDF.to_csv(f'{initDict.outDir}/sub-{subID}_day-{day}_modelvars.csv', index = False)
    sampleDF.to_csv(f'{initDict.outDir}/group_modelparams.csv', index = False)
    return

def unpackTask(taskDF):
    # Get choice attributes
    highChosen = np.array(taskDF.highChosen, dtype=bool)
    selectedStim = np.array(taskDF.choice)
    respIdx = selectedStim - 1
    
    # Get reversal attributes
    reverseTrial = np.array(taskDF.reverseTrial, dtype=bool)
    
    # Get stimulus attributes
    pWin = taskDF.Preward
    reward = taskDF.reward
    
    # Get session and trial lists
    sessID = np.array(taskDF.sessNo)
    trialNo = np.array(taskDF.trialNo)
    numTrials = len(taskDF)
    absTrials = np.arange(numTrials)
    
    # Get start of run
    runReset = np.zeros(numTrials)
    runReset[np.where(trialNo == 1)] = 1
    
    return dict2class(dict(runReset = runReset,
                           highChosen = highChosen,
                           selectedStim = selectedStim,
                           respIdx = respIdx,
                           reverseTrial = reverseTrial,
                           pWin = pWin,
                           reward = reward, 
                           sessID = sessID, 
                           trialNo = trialNo,
                           numTrials = numTrials,
                           absTrials = absTrials
                           ))
    
if __name__ == "__main__":
    fitModel()
