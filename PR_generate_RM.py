## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling/param_recovery')
import itertools
from defineModel import *
from utilities import *


# Generate data, loop across numSession lengths:
def runGenerate():
    numMaxDays = 6
    sessPerDays = 5
    for numSessions in np.arange(1,numMaxDays+1) * sessPerDays: 
        print(f'Simulation with {numSessions} sessions')
        genData(numSessions)
        
### Generate script
def initGenerate(numSessions):
    ###### Global task properties ######
    # Defining directories #
    homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling/param_recovery'
    if not os.path.exists(homeDir): 
        os.mkdir(homeDir)
    outDir =  homeDir + os.sep + 'Generate' + os.sep + 'Sessions_'+str(numSessions)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    initMod = ModelType()
    # Initialize the dictionary 
    initDict = dict2class(dict(outDir = outDir))
    return(initDict,initMod)

def genData(numSessions):
    # Number of different parameter values
    numParamEst = 10
    # Intialize structures and set up directories
    [initDict,initMod] = initGenerate(numSessions)
    # Number of simulations
    numIter = numParamEst**initMod.numParams
    # Set up range of model parameter values
    modelStruct = dict()
    # Generative parameters ()
    q_alpha, sm_beta  = initMod.genPriors(numParamEst)
    sub_params = []
    for param_perm in itertools.product(q_alpha, sm_beta):
        sub_params.append(param_perm)

    for subID in np.arange(numIter):
        # Specify model parameter for the current sim
        qA = sub_params[subID][0]
        smB = sub_params[subID][1]
        # Store the generation parameters
        genParams = genParamContain().instr_deltaLearner_genParams([qA, smB])
        # Set up the task parameters
        initDict = initTask(subID+1, initDict, numSessions)

        # Loop through sessions and t rials
        for sI in np.arange(initDict.numSessions):
            # Initialize the Q value
            qval = np.ones(2,dtype=float) * np.mean(np.append(initDict.outMag/100 , 0))
            sessionInfo = initDict.sessionInfo[sI]
            # Initialize the reverseStatus to False (need participant to get 4 continuous correct)
            reverseStatus = False
            for tI in range(initDict.trialsPerSess):
                # Initialize trials
                initTrial(tI, initDict, sessionInfo)
                # Run model simulation (make choice)
                [respIdx, _] = initMod.actor(qval, smB)
                initDict.sessionInfo[sI].sessionResponses.respKey[tI] = respIdx
                initDict.sessionInfo[sI].Qvalue[:,tI] = qval
                # Compute outcome 
                reward = computeOutcome(tI, initDict, sessionInfo, respIdx) 
                # Update action value according to the learning rule
                if (~np.isnan(reward)):
                    [qval[respIdx],_] = initMod.learner(qval[respIdx], qA, reward)
                # Compute reversal 
                reverseStatus = computeReversal(tI, initDict, sessionInfo, reverseStatus)
        # Store simulations
        modelStruct = dict2class(dict(genParams = genParams))
        # Convert data
        outPack = convertSave(initDict, modelStruct)
        # Save data (for current session)
        save_obj(outPack, initDict.outDir + os.sep + 'sim_' + str(initDict.subID))
        # Save as .mat files (for current session)
    return

def stimParam(initDict):
    # For the first axis, indices of 0 = stim1 and 1 = stim2
    pWin = np.empty((2,initDict.trialsPerSess), dtype=float)
    isHigh = np.empty((2,initDict.trialsPerSess), dtype=bool)
    isSelected = np.empty((2,initDict.trialsPerSess), dtype=float)
    isWin = np.empty((2,initDict.trialsPerSess), dtype=float)
    outMag = np.empty((2,initDict.trialsPerSess), dtype=float)
    return dict(pWin=pWin,
                isHigh=isHigh,
                isSelected=isSelected,
                isWin=isWin,
                outMag=outMag)
            
def initTask(subID, initDict, numSessions):
    # Specify task parameters
    pWinHigh = 0.65
    pWinLow = 0.35
    outMag = np.array([10,20,30])
    pReversal = 0.25
    # Set up the session-wise design
    trialsPerSess = 60
    # Flatten as dict2class    
    initDict.__dict__.update({
            'pWinHigh': pWinHigh,
            'pWinLow': pWinLow,
            'outMag': outMag,
            'pReversal': pReversal, 
            'subID':subID,
            'numSessions':numSessions,
            'trialsPerSess':trialsPerSess})
    sessionInfo = np.empty(initDict.numSessions, dtype=object)
    for sI in np.arange(numSessions):
        # Initialize (before first reversal) which stim is p(high)
        stim1_high = np.random.binomial(1, 0.5, 1).astype(bool)[0]
        # Store whether the good (pWinHigh) option was chosen
        highChosen = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Store which stim is the selected stim
        selectedStim = np.zeros(initDict.trialsPerSess,dtype=int)
        # Store whether reversals are possible on trial tI
        reverseStatus = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Store whether a reversal occurred on trial tI
        reverseTrial = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Initialize timing containers
        sessionResponses = Responses(initDict)
        # Initialize stim attribute containers
        stimAttrib = dict2class(stimParam(initDict))
        # Initialize payout container
        payOut = np.zeros(initDict.trialsPerSess,dtype=float)
        # Initialized the learnt values for instrumental actions (Q)
        Qvalue = np.empty([2, initDict.trialsPerSess],dtype=float)
        # Flatten into class object
        sessionInfo[sI] = dict2class(dict(stim1_high=stim1_high,
                                       highChosen=highChosen,
                                       selectedStim=selectedStim,
                                       reverseStatus=reverseStatus,
                                       reverseTrial=reverseTrial,
                                       sessionResponses=sessionResponses,
                                       stimAttrib=stimAttrib,
                                       payOut=payOut,
                                       Qvalue=Qvalue))
    initDict.__dict__.update({'sessionInfo':sessionInfo})
    return(initDict)



def computeOutcome(tI, initDict, sessionInfo, respIdx):
     # Draw win and loss magnitudes
     outMag = np.random.choice(initDict.outMag)
     # Determine which stim was chosen
     if (respIdx == 0):
         sessionInfo.selectedStim[tI] = 1
         pWin = sessionInfo.stimAttrib.pWin[respIdx,tI]
         isWin = np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respIdx == 1):
         sessionInfo.selectedStim[tI] = 2
         pWin = sessionInfo.stimAttrib.pWin[respIdx,tI]
         isWin = np.random.binomial(1,pWin,1).astype(bool)[0]          
     # Record stim attributes
     sessionInfo.stimAttrib.isSelected[respIdx, tI] = 1
     sessionInfo.stimAttrib.isWin[respIdx, tI] = isWin
     sessionInfo.stimAttrib.outMag[respIdx, tI] = outMag
     sessionInfo.stimAttrib.isSelected[1-respIdx, tI] = 0
     sessionInfo.stimAttrib.isWin[1-respIdx, tI] = np.nan
     sessionInfo.stimAttrib.outMag[1-respIdx, tI] = np.nan
     # Record whether they chose the high value option
     sessionInfo.highChosen[tI] = True if (pWin == initDict.pWinHigh) else False
     # Record the observed payOut
     sessionInfo.payOut[tI] = reward = outMag/100 if isWin else 0
     return reward


def initTrial(tI, initDict, sessionInfo):
    # Compute win probabilities for each stim
    if (sessionInfo.stim1_high):
        # Toggle which stim is high/low
        sessionInfo.stimAttrib.pWin[0, tI] = initDict.pWinHigh
        sessionInfo.stimAttrib.isHigh[0, tI] = True
        sessionInfo.stimAttrib.pWin[1, tI] = initDict.pWinLow
        sessionInfo.stimAttrib.isHigh[1, tI] = False
    else:
        # Toggle which stim is high/low
        sessionInfo.stimAttrib.pWin[0, tI] = initDict.pWinLow
        sessionInfo.stimAttrib.isHigh[0, tI] = False
        sessionInfo.stimAttrib.pWin[1, tI] = initDict.pWinHigh
        sessionInfo.stimAttrib.isHigh[1, tI] = True
    return



def computeReversal(tI, initDict, sessionInfo, reverseStatus):
    # No reversals in the first 4 trials of the task
    if (tI < 3):
        sessionInfo.reverseTrial[tI] = False
    # After the first 4 trials, reversals are possible
    if (tI >= 3):
        # Reversals are possible if 4 continuous correct responses
        if (np.all(sessionInfo.highChosen[tI-3:tI+1] == True)) and (np.all(np.diff(sessionInfo.selectedStim[tI-3:tI+1]) == 0)):
            reverseStatus = True
        # If 4 continuous incorrect responses, not sufficient learning. Reset reversalStatus
        if (np.all(sessionInfo.highChosen[tI-3:tI+1] == False)):
            reversalStatus = False
        # If reversals are possible
        sessionInfo.reverseStatus[tI] = reverseStatus
        # Store the reversal status of the trial
        if (reverseStatus):
            # Determine whether reversals occurs on this trials
            reverse = np.random.binomial(1, initDict.pReversal, 1).astype(bool)[0]
            if (reverse):
                # Execute high stim reversal
                sessionInfo.stim1_high = not sessionInfo.stim1_high
                sessionInfo.reverseTrial[tI] = True
                # Reset the reverseStatus
                reverseStatus = False
    return reverseStatus

    
# Execute    
if __name__ == "__main__":
    runGenerate()
        
