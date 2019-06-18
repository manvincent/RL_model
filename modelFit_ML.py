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
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/analysis/modelling')
from onsets import * 
from optimizer import *
from utilities import *
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/rev_py_v01')

### Recovery script
def initRecover():
    modelName = 'base_delta'
    homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/analysis'
    datDir =  '/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/rev_py_v01/Output'
    outDir = homeDir + os.sep + 'modelling' + os.sep + 'model_results'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    numParams = 2
    initDict = dict2class(dict(modelName = modelName,   
                           homeDir = homeDir,
                           datDir = datDir,
                           outDir = outDir))
    return(initDict)


def fitModel(fmri = True):    
    # Specify subject list
    # Specify experiment info
    numSubs = 7
    subList = np.arange(6,12+1)
    # Initialize the model fitting object
    initDict = initRecover()
    # Create structures for storing estimates across subjects
    sampleDF = pd.DataFrame()
    for subIdx, subID in enumerate(subList):
        # Load procedure
        inFile = 'sub' + str(subID) + '_sess1_data'
        # Load in data
        dataFile = load_obj(initDict.datDir + os.sep + inFile + '.pkl')
        # Unpack the task sessions
        taskData = unpackTask(dataFile)
        numTrials=len(taskData.respIdx) 
        # Print current fit details
        print(f'Subject No: {subIdx+1}, subID: {subID}')
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
        print(f'LR: {fitParams.alpha_i}')
        print(f'smB: {fitParams.beta_i}')        
        # Append fitted parameters to dataframe
        sampleDF = sampleDF.append({'subID': int(subID),
                                    'fit_alpha': fitParams.alpha_i,
                                    'fit_beta': fitParams.beta_i}, ignore_index=True)
        
        # Simulate using retrieved parameter to compute computational variables        
        RPE = np.zeros(numTrials, dtype = float) 
        Q = np.empty((2,numTrials), dtype = float)
        chosenQ = np.empty(numTrials, dtype = float)
        unchosenQ = np.empty(numTrials, dtype = float)
        simResp = np.empty(numTrials, dtype = float) 
        # Initialize qval
        qval = np.zeros(2, dtype = float)
        for tI in np.arange(numTrials):
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
        compVarDF.to_csv(f'{initDict.outDir}/s{subID}_modelvars.csv', index = False)    
         # Create fMRI onsets (fsl stype) 
        if fmri: 
            timeDF = pd.read_csv(f'{initDict.homeDir}/data/sub' + str(subID) + '_onsets.csv')
            expDF =  pd.read_csv(f'{initDict.homeDir}/data/sub' + str(subID) + '_data.csv')
            cueVar = ['stim1_Q','stim2_Q','chosenQ','unchosenQ']
            outVar = ['RPE','absRPE']
            createOnsets(subID, timeDF, expDF, compVarDF, cueVar, outVar)
    sampleDF.to_csv(f'{initDict.outDir}/group_modelparams.csv', index = False)    
    return 

def unpackTask(genTask):
    # Get choice attributes
    highChosen = np.hstack([genTask.sessionInfo[i].highChosen for i in np.arange(genTask.numSessions)])
    selectedStim = np.hstack([genTask.sessionInfo[i].selectedStim for i in np.arange(genTask.numSessions)])
    respIdx = np.array([])
    for i in np.arange(genTask.numSessions):
        sessionInfo = genTask.sessionInfo[i]
        sess_respIdx = np.array([0 if sessionInfo.stimAttrib.isSelected[0,tI] == 1 
                            else 1 if sessionInfo.stimAttrib.isSelected[1,tI] == 1 
                            else np.nan 
                            for tI in np.arange(genTask.trialInfo.trialsPerSess)])    
        respIdx = np.concatenate([respIdx, sess_respIdx])
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
