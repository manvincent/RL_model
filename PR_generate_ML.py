## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import os
os.chdir("C:\\Users\\Brenden\\OneDrive - California Institute of Technology\\PhD\\O'Doherty Lab\\RL_model")
import itertools
from defineModel import *
from utilities import *
from initTask import *


### Generate script
def initGenerate(numTrials):
    ###### Global task properties ######
    # Defining directories #
    homeDir = "C:\\Users\\Brenden\\OneDrive - California Institute of Technology\\PhD\\O'Doherty Lab\\RL_model\\param_recov"
    if not os.path.exists(homeDir): 
        os.mkdir(homeDir)
    outDir =  homeDir + os.sep + 'Generate' + os.sep + 'Trial_'+str(numTrials)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    numParams = 2
    initMod = ModelType()
    # Initialize the dictionary 
    initDict = dict2class(dict(outDir = outDir))
    return(initDict,initMod)

def genData(numTrials):
    # Number of different parameter values
    numParamEst = 10
    # Number of simulations 
    numIter = numParamEst**2
    # Intialize structures and set up directories
    [initDict,initMod] = initGenerate(numTrials)
    # Set up range of model parameter values
    modelStruct = dict()
    # Base instrumental parameters
    q_alpha = np.linspace(initMod.alpha_bounds[0],initMod.alpha_bounds[1],numParamEst) # Q-learning alpha (learning rate)
    sm_beta = np.linspace(initMod.beta_bounds[0],initMod.beta_bounds[1],numParamEst)
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
        initDict = initTask(expInfo)

        # Initialize the Q value
        qval = np.zeros(2,dtype=float)
        # Loop through sessions and t rials
        for sI in np.arange(initDict.numSessions):
            sessionInfo = initDict.sessionInfo[sI]
            # Initialize the reverseStatus to False (need participant to get 4 continuous correct)
            reverseStatus = False
            for tI in range(initDict.trialInfo.trialsPerSess):
                # Initialize sessions
                initSessions(expInfo,tI, sessionInfo)
                # Run model simulation (make choice)
                [respIdx, _] = initMod.actor(qval, smB)
                initDict.sessionInfo[0].sessionResponses.respKey[tI] = respIdx
                initDict.sessionInfo[0].Qvalue[:,tI] = qval
                # Compute outcome 
                reward = computeOutcome(tI, 
                    initDict.numSessTrials, 
                    initDict.pWinHgih,
                    initDict.pWinMed[sessionInfo.sessIsVol],
                    taskInfo.pWinLow,
                    vol=sessionInfo.sessIsVol)
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


# Generate data, loop across numTrial lengths:
expInfo = {'SubNo': 999,
               'Version': 'test',
               'Condition': 'money',
               'Modality': 'fMRI',
               'doInstruct': True,
               'doPract': True,
               'doTask': True}
for numTrials in np.arange(1,16) * 10:
    genData(numTrials)
