#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:09:24 2019

@author: vman
"""
import os
import numpy as np
import pandas as pd

# Ensure that relative paths start from the same directory as this script
homeDir = 'path/to/project''
analysisDir = homeDir + 'analysis/'
if not os.path.exists(analysisDir + 'data'):
    os.mkdir(analysisDir + 'data')
taskDir = homeDir + 'rev_py_v01/'
datDir = taskDir + 'Output/'

# Import custom modules
os.chdir(taskDir)
from config import *

# Specify experiment info
numSubs = 23
subList = np.arange(numSubs)+1
numSess = 1

def parseData(subList, numSess):
    group_DF = []
    # Iterate through subjects
    for subID in subList:
        # Check if subject data exists
        if os.path.exists(datDir + 'sub' + str(subID) + '_sess1_data.pkl'):
            # Create dataframe
            subDF = pd.DataFrame()
            # Iterate through sessions
            for sessIdx, sessID in enumerate(np.arange(numSess)+1):
                # Load in subject data
                subDat = load_obj(datDir + 'sub' + str(subID) + '_sess' + str(sessID) + '_data.pkl')
                # Load up session data
                sessDat = subDat.sessionInfo[sessIdx]
                # Interpolate missing data if applicable
                if not hasattr(sessDat, 'stimAttrib'):
                    interpDesign(sessDat, subDat.trialInfo.trialsPerSess, subDat.pWinHigh, subDat.pWinLow)
                # Create session dataframe
                sessDF = pd.DataFrame(columns=['sessNo',
                                               'trialNo',
                                               'block',
                                               'blockTrialNo',
                                               'trialClass',
                                               'blockBin',
                                               'stim1_High',
                                               'stim2_High',
                                               'stimHigh',
                                               'stim1_left',
                                               'leftStim',
                                               'rightStim',
                                               'stim1_pWin',
                                               'stim2_pWin',
                                               'reverseStatus',
                                               'reverseTrial',
                                               'responseKey',
                                               'response_LR',
                                               'selected_stim1',
                                               'selected_stim2',
                                               'response_stimID',
                                               'RT',
                                               'highChosen',
                                               'stim1_isWin',
                                               'stim2_isWin',
                                               'isWin',
                                               'payOut',
                                               'accum_payOut',
                                               'abs_outMag'])
                # Populate design info
                # Stim attributes
                sessDF.stim1_High = sessDat.stimAttrib.isHigh[0,:]
                sessDF.stim2_High = sessDat.stimAttrib.isHigh[1,:]
                sessDF.stimHigh = sessDF.stim1_High.apply(lambda x: 1 if x else 2)
                sessDF.stim1_left = sessDat.stim1_left
                sessDF.leftStim = sessDF.stim1_left.apply(lambda x: 1 if x else 2)
                sessDF.rightStim = sessDF.stim1_left.apply(lambda x: 2 if x else 1)
                sessDF.stim1_pWin = sessDat.stimAttrib.pWin[0,:]
                sessDF.stim2_pWin = sessDat.stimAttrib.pWin[1,:]
                # Reversal attributes
                sessDF.reverseStatus = sessDat.reverseStatus
                sessDF.reverseTrial = sessDat.reverseTrial
                # Response attributes
                sessDF.responseKey = sessDat.sessionResponses.respKey
                sessDF.response_LR = sessDF.responseKey.apply(lambda x: 'left' if x==1 else ('right' if x==4 else np.nan))
                sessDF.selected_stim1 = sessDat.stimAttrib.isSelected[0,:]
                sessDF.selected_stim2 = sessDat.stimAttrib.isSelected[1,:]
                sessDF.response_stimID = sessDF.selected_stim1.apply(lambda x: 1 if x==1 else (2 if x==0 else np.nan))
                sessDF.RT = sessDat.sessionResponses.rt
                # Outcome attributes
                sessDF.highChosen = sessDat.highChosen
                sessDF.stim1_isWin = sessDat.stimAttrib.isWin[0,:]
                sessDF.stim2_isWin = sessDat.stimAttrib.isWin[1,:]
                sessDF.isWin = [1 if (sessDat.stimAttrib.isWin[0,tI] == 1) or (sessDat.stimAttrib.isWin[1,tI] == 1) else (0 if (sessDat.stimAttrib.isWin[0,tI] == 0) or (sessDat.stimAttrib.isWin[1,tI] == 0) else np.nan) for tI in np.arange(subDat.trialInfo.trialsPerSess)]
                sessDF.payOut = sessDat.payOut
                sessDF.payOut[sessDF.payOut==0] = np.nan
                sessDF.accum_payOut = pd.Series.cumsum(sessDF.payOut)
                sessDF.abs_outMag = np.abs(sessDF.payOut)
                # Session and trial number
                sessDF.sessNo = sessID
                sessDF.trialNo = np.arange(len(sessDat.sessionOnsets.tPreFix))+1
                sessDF.blockTrialNo, sessDF.block =  blockCounter(sessDF.reverseTrial) # Trial number before reversal
                sessDF.trialClass, sessDF.blockBin = trialClassifier(sessDF.blockTrialNo,
                                                                     sessDF.block,
                                                                     thres = 0.5,
                                                                     binWidth = 5) # Classify within-block trials as 'early' vs 'late'

                # Concatenate session DF (if multiple sessions)
                subDF = pd.concat([subDF,sessDF],axis=0)
                # Onsets
                onsetDF = pd.DataFrame(columns=['tPreFix','tStim','tResp','tOut','tPostFix'])
                onsetDF.tPreFix = sessDat.sessionOnsets.tPreFix
                onsetDF.tStim = sessDat.sessionOnsets.tStim
                onsetDF.tResp = sessDat.sessionOnsets.tResp
                onsetDF.tOut= sessDat.sessionOnsets.tOut
                onsetDF.tPostFix = sessDat.sessionOnsets.tPostFix
                # Output onset DF
                onsetDF.to_csv(analysisDir + 'data' + os.sep + 'sub' + str(subID) + '_onsets.csv', na_rep=np.nan, index=False)
                # Append to group dataframe
                group_DF.append(subDF)

            # Output subject data
            subDF['instructCond'] = subDat.instructCond if hasattr(subDat,'instructCond') else 'hm'
            subDF['subID'] = subID
            subDF.to_csv(analysisDir + 'data' + os.sep + 'sub' + str(subID) + '_data.csv', na_rep=np.nan, index=False)

        # Create group dataframe
        groupDF = pd.concat(group_DF,axis=0)
    # Output group data
    groupDF.to_csv(analysisDir + 'data' + os.sep + 'group_data.csv', na_rep=np.nan, index=False)
    return

# Function to interpolate data for subjects with missing data
# input data is sessData
def interpDesign(data, trialsPerSess, pWinHigh, pWinLow):
    isSelected = np.empty((2,trialsPerSess),dtype=float)
    isHigh = np.empty((2,trialsPerSess),dtype=bool)
    pWin = np.empty((2,trialsPerSess),dtype=float)
    isWin = np.empty((2,trialsPerSess),dtype=float)
    for tI in np.arange(trialsPerSess):
        if (data.stim1_left[tI]):
            respStimIdx = 0 if (data.sessionResponses.respKey[tI] == 1) else 1
        elif (not data.stim1_left[tI]):
            respStimIdx = 1 if (data.sessionResponses.respKey[tI] == 1) else 0
        # isSelected field
        isSelected[respStimIdx,tI] = 1
        isSelected[1-respStimIdx,tI] = 0
        isSelected[:,tI] = np.nan if (np.isnan(data.sessionResponses.respKey[tI])) else isSelected[:,tI]
        # isHigh field
        if (data.highChosen[tI]):
            isHigh[respStimIdx,tI] = True
            isHigh[1-respStimIdx,tI] = False
        else:
            isHigh[respStimIdx,tI] = False
            isHigh[1-respStimIdx,tI] = True
        # pWin field
        pWin[0,tI] = pWinHigh if (isHigh[0,tI]) else pWinLow
        pWin[1,tI] = pWinHigh if (isHigh[1,tI]) else pWinLow
        # isWin field
        isWin[1-respStimIdx,tI] = np.nan
        isWin[respStimIdx,tI] = 1 if (np.sign(data.payOut[tI]) > 0) else 0
        isWin[:,tI] = np.nan if (np.isnan(data.sessionResponses.respKey[tI])) else isWin[:,tI]
    # Wrap into 'stimAttrib' class object
    stimAttrib = dict2class(dict(pWin=pWin,
                                 isHigh=isHigh,
                                 isSelected=isSelected,
                                 isWin=isWin))
    # Update data object with this previously missing object (stimAttrib)
    data.__dict__.update({'stimAttrib':stimAttrib})
    return

# Compute the within-block (pre-reversal trial indices)
def blockCounter(reverseTrial):
    output = np.array([])
    block = np.array([])
    trialCounter = 1
    blockCounter = 1
    for i in np.arange(len(reverseTrial)):
        # Update the output array
        output = np.append(output, trialCounter)
        block = np.append(block, blockCounter)
        if reverseTrial[i] == True:
            trialCounter = 1
            blockCounter += 1
        else:
            trialCounter += 1
    return(output, block)

# Classify trials as being 'early' or late' wrt pre-reversal blocks
def trialClassifier(blockTrialNo, block, thres, binWidth):
    # Classifies trials as early or late
    label = np.array([])
    for blockIdx in np.unique(block):
        numBlockTrials = len(block[block == blockIdx])
        numClassTrials = np.ceil(thres * numBlockTrials).astype(int)
        # Create 'trial type' label
        blockLabel = np.full([numBlockTrials],'undefined')
        blockLabel[:numClassTrials] = 'early'
        blockLabel[-numClassTrials:] = 'late'
        label = np.append(label,blockLabel)
    # Creates bins within each block
    blockTrialCount = np.array(block.groupby(block).count()) # Get block trial count
    blockNumBins = np.floor(blockTrialCount/binWidth) #
    binLabel = np.array([])
    for blockIdx in np.arange(len(np.unique(block))):
        blockBinLabel = np.sort(np.tile(np.arange(blockNumBins[blockIdx])+1, binWidth))
        if (len(blockBinLabel) != blockTrialCount[blockIdx]):
            blockBinLabel = np.append(blockBinLabel, np.tile(blockNumBins[blockIdx]+1, blockTrialCount[blockIdx] - len(blockBinLabel)))
        binLabel = np.append(binLabel, blockBinLabel)
    return(label, binLabel)



# Parse data
if __name__ == "__main__":
    parseData(subList, numSess)
