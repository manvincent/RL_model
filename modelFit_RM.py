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
    modelName = 'base_delta'
    homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis'
    datDir =  f'{homeDir}/data'
    outDir = f'{homeDir}/modelling/model_results'
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
    subList = np.array([3])
    #numDays = np.array([6,4,3,1])
    numDays = np.array([6])
    modality = 'fmri' # can be behav (for pilot 2) or fmri 
    # Initialize the model fitting object
    initDict = initRecover()
    # Create structures for storing estimates across subjects
    sampleDF = pd.DataFrame()
    for subIdx, subID in enumerate(subList):
        # Load procedure
        inFile = f'sub{subID}_data.csv'
        # Load in data
        subDF = pd.read_csv(f'{initDict.datDir}/{modality}/{inFile}')
        for day in np.arange(4,numDays[subIdx])+1:
            dayDF = subDF[subDF.dayID == day]
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
                    reward = taskData.payOut[tI]
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
    selectedStim = np.array(taskDF.response_stimID)
    respIdx = selectedStim - 1
    
    # Get reversal attributes
    reverseStatus = np.array(taskDF.reverseStatus, dtype=bool)
    reverseTrial = np.array(taskDF.reverseTrial, dtype=bool)
    
    # Get stimulus attributes
    pWin = np.array([taskDF.stim1_pWin, taskDF.stim2_pWin])
    isHigh = np.array([taskDF.stim1_High, taskDF.stim2_High], dtype=bool)
    isSelected = np.array([taskDF.selected_stim1, taskDF.selected_stim2])
    isWin = np.array([taskDF.stim1_isWin, taskDF.stim2_isWin])
    
    # Get outcome attributes
    outMag = minmax(np.array(taskDF.outMag))
    
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
                           reverseStatus = reverseStatus,
                           reverseTrial = reverseTrial,
                           pWin = pWin,
                           isHigh = isHigh,
                           isSelected = isSelected,
                           isWin = isWin,
                           payOut = outMag, 
                           sessID = sessID, 
                           trialNo = trialNo,
                           numTrials = numTrials,
                           absTrials = absTrials
                           ))
def minmax(array): 
    return (array - min(array)) / (max(array) -min(array))
    
if __name__ == "__main__":
    fitModel()
