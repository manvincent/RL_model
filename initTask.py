from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os
from config import *

def initTask(expInfo):
   ###### Task parameters properties ######
    # task properties
    numSessions = 4
    numTrials = 240
    # trial timing
    def trialParam(numSessions, numTrials):
        maxRT = 3
        isiMinTime = 1.25
        isiMaxTime = 1.75
        fbMinTime = 1.25
        fbMaxTime = 1.75
        delivTime = 2.5
        if (expInfo.Modality == 'fMRI'):
            TR = 1.12 # Delete first 4 volumes
            disDaqTime = 4 * TR
        elif (expInfo.Modality == 'behaviour'):
            disDaqTime = 0
        minJitter = 1
        maxJitter = 2
        trialsPerSess = numTrials // numSessions
        return dict(maxRT=maxRT,
                    isiMinTime=isiMinTime,
                    isiMaxTime=isiMaxTime,
                    fbMinTime=fbMinTime,
                    fbMaxTime=fbMaxTime,
                    delivTime=delivTime,
                    disDaqTime=disDaqTime,
                    minJitter=minJitter,
                    maxJitter=maxJitter,
                    trialsPerSess=trialsPerSess)
    trialInfo = dict2class(trialParam(numSessions, numTrials))
    # Win probabilities

    def taskParam():
        subID = int()
        instructCond = str()
        numStims = 3
        pWinHigh = 0.8
        pWinLow = 0.2
        pWinMed = np.array([[0.35, 0.65], [0.3, 0.5, 0.7]], dtype=object)
        outMag = np.array([25, 50, 75])
        # Set up possible stim combinations
        stimComb = np.array([[0,1],[1,2],[0,2]])
        return dict(subID=subID,
                    instructCond=instructCond,
                    numStims=numStims,
                    pWinHigh=pWinHigh,
                    pWinLow=pWinLow,
                    pWinMed=pWinMed,
                    outMag=outMag,
                    stimComb=stimComb)
    taskInfo = dict2class(taskParam())
    taskInfo.__dict__.update({'trialInfo': trialInfo,
                              'numSessions': numSessions,
                              'numTrials': numTrials})
    ###### Setting up the display structure #######

    def dispParam(expInfo):
        xRes = 1600
        yRes = 1200
        screenColor=[-0.5,-0.5,-0.5]
        screenColSpace='rgb'
        screenPos=(0, 0)
        screenUnit='norm'
        screenWinType='pyglet'
        if (expInfo.Version == 'debug'):
            screenScaling = 0.5
            screen = visual.Window(color=screenColor,
                                   colorSpace=screenColSpace,
                                   size=(xRes * screenScaling, yRes * screenScaling),
                                   pos=screenPos,
                                   units=screenUnit,
                                   winType=screenWinType,
                                   fullscr=False,
                                   screen=0,
                                   allowGUI=True)
        elif (expInfo.Version == 'test'):
            screenScaling = 1
            screen = visual.Window(color=screenColor,
                                   colorSpace=screenColSpace,
                                   size=(xRes * screenScaling, yRes * screenScaling),
                                   pos=screenPos,
                                   units=screenUnit,
                                   winType=screenWinType,
                                   fullscr=True,
                                   screen=1,
                                   allowGUI=False)
        monitorX = screen.size[0]
        monitorY = screen.size[1]
        fps = screen.getActualFrameRate(nIdentical=10,
                                        nMaxFrames=100,
                                        nWarmUpFrames=10,
                                        threshold=1)
        textFont = 'Helvetica'
        imageSize = 0.5
        imagePosL = [-0.5,0]
        imagePosR = [0.5,0]
        dispInfo = dict2class(dict(screenScaling=screenScaling,
                                monitorX=monitorX,
                                monitorY=monitorY,
                                fps=fps,
                                textFont=textFont,
                                imageSize=imageSize,
                                imagePosL=imagePosL,
                                imagePosR=imagePosR))
        return dispInfo, screen
    [dispInfo, screen] = dispParam(expInfo)

    # Set up python objects for all generic task objects

    # Start loading images
    loadScreen = visual.TextStim(screen,
                                 text="Loading...",
                                 font=dispInfo.textFont,
                                 pos=screen.pos,
                                 height=0.1,
                                 color='white')
    loadScreen.setAutoDraw(True)
    screen.flip()
    # display 'save' screen
    saveScreen = visual.TextStim(screen,
                                 text="Saving...",
                                 font=dispInfo.textFont,
                                 pos=screen.pos,
                                 height=0.1,
                                 color='white')
    # Keyboard info
    keyInfo = dict2class(keyConfig())

    # Stimuli
    stims = np.empty(taskInfo.numStims, dtype=object)
    for idx in np.arange(taskInfo.numStims):
        stims[idx] = TrialObj(taskInfo,
                                type="stim",
                                pathToFile=f'{expInfo.stimDir}/fract{idx}',
                                mag=0)

    # Outcome
    outGain = np.empty(len(taskInfo.outMag), dtype=object)
    for idx, mag in enumerate(taskInfo.outMag):
        outGain[idx] = TrialObj(taskInfo,
                                type='out',
                                pathToFile=f'{expInfo.stimDir}/cb_{expInfo.sub_cb}/{expInfo.Condition}',
                                mag=mag)
    outNoGain = TrialObj(taskInfo, type='noOut',
                                pathToFile=f'{expInfo.stimDir}/cb_{expInfo.sub_cb}/noGain',
                                mag=0)

    # Fixations
    startFix = visual.TextStim(screen,
                               text="+",
                               font=dispInfo.textFont,
                               pos=[0, 0],
                               height=0.15,
                               color='white',
                               wrapWidth=1.8)
    endFix = visual.TextStim(screen,
                             text="+",
                             font=dispInfo.textFont,
                             pos=screen.pos,
                             height=0.15,
                             color='white',
                             wrapWidth=1.8)

    expEndFix = visual.TextStim(screen,
                             text="+",
                             font=dispInfo.textFont,
                             pos=screen.pos,
                             height=0.15,
                             color='red',
                             wrapWidth=1.8)
    # Stimuli
    leftStim = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosL)
    rightStim = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosR)
    # Responses (ISI)
    leftResp = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosL)
    rightResp = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosR)
    # Outcomes
    leftOut = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosL)
    rightOut = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosR)
    # Reward delivery
    leftDeliv = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosL)
    rightDeliv = visual.ImageStim(win=screen,
                                size=dispInfo.imageSize,
                                pos=dispInfo.imagePosR)
    # Initialize special messages
    waitExp = visual.TextStim(screen,
                              text="Please get ready. Waiting for experimenter...",
                              font=dispInfo.textFont,
                              pos=screen.pos,
                              height=0.15,
                              color='white',
                              wrapWidth=1.8)
    readyExp = visual.TextStim(screen,
                               text="Please get ready. Press " + keyInfo.instructDone + " to start",
                               font=dispInfo.textFont,
                               pos=screen.pos,
                               height=0.15,
                               color='white',
                               wrapWidth=1.8)
    scanPulse = visual.TextStim(screen,
                                text="Waiting for the scanner...",
                                font=dispInfo.textFont,
                                pos=screen.pos,
                                height=0.15,
                                color='white',
                                wrapWidth=1.8)
    noRespErr = visual.TextStim(screen,
                                text="Please respond faster. This trial has been cancelled.",
                                font=dispInfo.textFont,
                                pos=screen.pos,
                                height=0.07,
                                color='white',
                                wrapWidth=1.8)
    # Initialize pract start/end screens
    practStart = visual.TextStim(screen,
                              text="Get ready for some practice.\n\nThe practice has no impact on your overall money.\n\nTry to choose both images while practicing!",
                              font=dispInfo.textFont,
                              pos=screen.pos,
                              height=0.15,
                              color='white',
                              wrapWidth=1.8)
    practEnd = visual.TextStim(screen,
                               text="End of practice. Get ready for the task.\n\nYou will start playing for real money.",
                               font=dispInfo.textFont,
                               pos=screen.pos,
                               height=0.15,
                               color='white',
                               wrapWidth=1.8)
    # ITI object
    ITI = core.StaticPeriod(screenHz=dispInfo.fps, win=screen, name='ITI')
    # Wrap objects into dictionary
    taskObj = dict2class(dict(screen=screen,
                           loadScreen=loadScreen,
                           saveScreen=saveScreen,
                           stims=stims,
                           outGain=outGain,
                           outNoGain=outNoGain,
                           startFix=startFix,
                           endFix=endFix,
                           expEndFix=expEndFix,
                           leftStim=leftStim,
                           rightStim=rightStim,
                           leftResp=leftResp,
                           rightResp=rightResp,
                           leftOut=leftOut,
                           rightOut=rightOut,
                           leftDeliv=leftDeliv,
                           rightDeliv=rightDeliv,
                           waitExp=waitExp,
                           readyExp=readyExp,
                           scanPulse=scanPulse,
                           noRespErr=noRespErr,
                           practStart=practStart,
                           practEnd=practEnd,
                           ITI=ITI))

    # Initialize task variables
    taskInfo = initSessions(expInfo, taskInfo, numSessions)
    # Close loading screen
    loadScreen.setAutoDraw(False)
    return screen, dispInfo, taskInfo, taskObj, keyInfo



class TrialObj(object):
    def __init__(self, taskInfo, type, pathToFile, mag):
        # Static object parameters
        if (type == "stim"):
            self.path = f'{pathToFile}.png'
            self.respPath = f'{pathToFile}_resp.png'
            # Initialize design containers
            self.pWin = float()
        elif (type == "out"):
            self.path = f'{pathToFile}_gain_{mag}.png'
            self.delivPath = f'{pathToFile}_deliv_{mag}.png'
        elif (type == "noOut"):
            self.path = self.delivPath = f'{pathToFile}.png'


class Onsets(object):
    def __init__(self,taskInfo):
        self.tPreFix = np.empty(taskInfo.trialInfo.trialsPerSess)
        self.tStim = np.empty(taskInfo.trialInfo.trialsPerSess)
        self.tResp = np.empty(taskInfo.trialInfo.trialsPerSess)
        self.tOut = np.empty(taskInfo.trialInfo.trialsPerSess)
        self.tPostFix = np.empty(taskInfo.trialInfo.trialsPerSess)

class Responses(object):
    def __init__(self,taskInfo):
        self.respKey = np.empty(taskInfo.trialInfo.trialsPerSess)
        self.rt = np.empty(taskInfo.trialInfo.trialsPerSess)


def computeOutcome(taskInfo, numSessTrials, highVal, medVal, lowVal, vol):
    # Create bins of variable sizes within run
    numReversal = 1 if vol == 0 else 2
    numBins = numReversal + 1
    meanBinLength = numSessTrials // numBins
    varBinLength = 10 if vol ==0 else 5
    def tReverse(mean, var):
        return np.random.randint(mean - var, mean + var)
    # Specify how many trials per bin
    if vol == 0:
        firstReversal = tReverse(meanBinLength, varBinLength)
        binLengths = np.array([firstReversal, (numSessTrials - firstReversal)])
    elif vol == 1:
        firstReversal = tReverse(meanBinLength, varBinLength)
        secondReversal = tReverse(meanBinLength*2, varBinLength)
        binLengths = np.array([firstReversal,
                            numSessTrials - firstReversal - (numSessTrials - secondReversal),
                            (numSessTrials - secondReversal)])
    # Assign values to bins according to pWinHigh and pWinLow vectors above
    highValArray = np.array([highVal-0.05, highVal, highVal+0.05])
    lowValArray = np.array([lowVal-0.05, lowVal, lowVal+0.05])
    highLow_valArray = [highValArray, lowValArray]
    # Assign values according to pWinMed for high and low Volatility
    # Initialize which bin starts high
    isHigh = np.random.binomial(1,0.5)
    medIsHigh = np.random.binomial(1,0.5)
    binProbs = np.zeros([taskInfo.numStims, numBins], dtype=float)
    for b in np.arange(numBins):
        binProbs[0,b] = np.random.choice(highLow_valArray[isHigh]) if b % 2 != 0 else np.random.choice(highLow_valArray[1 - isHigh])
        binProbs[2,b] = np.random.choice(highLow_valArray[1 - isHigh]) if b % 2 != 0 else  np.random.choice(highLow_valArray[isHigh])
    if vol == 0:
        binProbs[1,:] = np.random.permutation(medVal)
    elif vol == 1:
        binProbs[1,:] = np.flip(medVal) if np.random.choice([True, False]) else medVal
    # bins are stored as 3 (stims 1,2,3) x n (number of blocks)

    # Broadcast bin probabilities onto trials
    trialProbs = np.empty([taskInfo.numStims, numSessTrials], dtype=float)
    for stim in np.arange(taskInfo.numStims):
        trialProbs[stim,:] = np.repeat(binProbs[stim], numSessTrials//numBins)

    # Compute trial-wise outcomes based on bin lengths and probabilities
    outcomes = np.empty([taskInfo.numStims, numSessTrials], dtype=int)
    magnitudes = np.empty([taskInfo.numStims, numSessTrials], dtype=int)
    for stim in np.arange(taskInfo.numStims):
        stimOuts = []
        stimMags = []
        for b in np.arange(numBins):
            # Initialize outcome array based on bin lengths
            outArray = np.zeros(binLengths[b])
            magArray = np.zeros(binLengths[b])
            # Popluate outcome array based on pWin
            outArray[:int(binProbs[stim,b]*binLengths[b])] = 1
            # Populate magnitude array
            numBinWins = len(np.where(outArray == 1)[0])
            magArray[np.where(outArray == 1)[0][0 : numBinWins-numBinWins%3]] = np.random.permutation(np.repeat(taskInfo.outMag,numBinWins // 3))
            magArray[np.where(outArray == 1)[0][numBinWins-numBinWins%3 : numBinWins]] = taskInfo.outMag[1]
            # randomize and append to correct stimulus dimension
            randomize = np.random.permutation(np.arange(len(outArray)))
            stimOuts.append(outArray[randomize])
            stimMags.append(magArray[randomize])
        outcomes[stim,:] = np.concatenate(stimOuts)
        magnitudes[stim,:] = np.concatenate(stimMags)
    # outcomes are stored as 3 (stims 1,2,3) x 60 (session trials)
    return outcomes, magnitudes, binProbs, binLengths, trialProbs

def initSessions(expInfo, taskInfo, numSessions):
    # Set up the session-wise design
    sessionInfo = np.empty(taskInfo.numSessions, dtype=object)
    # Set up which sessions are high vs low volatility
    highVolSess = np.random.permutation(np.repeat([0,1],numSessions//2))
    # Set up for each session (scan run)
    for sI in range(taskInfo.numSessions):
        numSessTrials = taskInfo.trialInfo.trialsPerSess
        # Volatility
        sessIsVol = highVolSess[sI]
        # Initialize which stim is p(high), p(med), p(low)
        # prob and outcome trajectory index, should be broadcast onto stim0, stim1, stim2
        stim_valAssign = np.random.permutation(np.arange(taskInfo.numStims))

        # Determine the outcomes in this session
        outcomes, magnitudes, binProbs, binLengths, trialProbs = computeOutcome(taskInfo,
                                                                    numSessTrials,
                                                                    taskInfo.pWinHigh,
                                                                    np.array(taskInfo.pWinMed[sessIsVol]),
                                                                    taskInfo.pWinLow,
                                                                    vol=sessIsVol)

        # Determine which stims are presented on each trial
        stimList = []
        for b in np.arange(len(binLengths)):
            binStims = np.repeat(taskInfo.stimComb, binLengths[b]//taskInfo.numStims, axis=0)
            np.random.shuffle(binStims)
            if binLengths[b] % taskInfo.numStims != 0:
                for res in np.arange(binLengths[b] % taskInfo.numStims):
                    binStims = np.vstack((binStims, taskInfo.stimComb[np.random.randint(0,3)]))
            stimList.append(binStims)
        stimOrder = np.concatenate(stimList)

        # Broadcast probabilities and outcomes onto stims for each trial
        trialProb = np.empty([numSessTrials, 2], dtype=float)
        trialOut = np.empty([numSessTrials, 2], dtype=bool)
        trialMag = np.empty([numSessTrials, 2], dtype=int)
        for t in np.arange(numSessTrials):
            for stimIdx in np.arange(2):
                trialProb[t,stimIdx] = trialProbs[np.where(stimOrder[t,stimIdx]==stim_valAssign),t]
                trialOut[t,stimIdx] =  outcomes[np.where(stimOrder[t,stimIdx]==stim_valAssign),t]
                trialMag[t, stimIdx] = magnitudes[np.where(stimOrder[t,stimIdx]==stim_valAssign),t]


        # Trial design randomisations
        itiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.minJitter,
                                     taskInfo.trialInfo.maxJitter,
                                     numSessTrials))
        isiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.isiMinTime,
                                     taskInfo.trialInfo.isiMaxTime,
                                     numSessTrials))
        fbDur = np.random.permutation(np.linspace(taskInfo.trialInfo.fbMinTime,
                                     taskInfo.trialInfo.fbMaxTime,
                                     numSessTrials))
        # Store whether the good (pWinHigh) option was chosen
        highChosen = np.zeros(numSessTrials,dtype=bool)
        # Store which stim is the selected stim
        selectedStim = np.zeros(numSessTrials,dtype=int)
        # Initialize timing containers
        sessionOnsets = Onsets(taskInfo)
        sessionResponses = Responses(taskInfo)
        # Initialize stim attribute containers
        def stimParam(taskInfo):
            # For the first axis, indices of 0 = stim1 and 1 = stim2
            pWin = np.empty((taskInfo.numStims,numSessTrials), dtype=float)
            isSelected = np.empty((taskInfo.numStims,numSessTrials), dtype=float)
            isWin = np.empty((taskInfo.numStims,numSessTrials), dtype=float)
            outMag = np.empty((taskInfo.numStims,numSessTrials), dtype=float)
            return dict(pWin=pWin,
                        isSelected=isSelected,
                        isWin=isWin,
                        outMag=outMag)
        stimAttrib = dict2class(stimParam(taskInfo))
        # Initialize payout container
        payOut = np.zeros(taskInfo.trialInfo.trialsPerSess,dtype=float)
        # Flatten into class object
        sessionInfo[sI] = dict2class(dict(numSessTrials=numSessTrials,
                                          sessIsVol=sessIsVol,
                                          stim_valAssign=stim_valAssign,
                                          outcomes=outcomes,
                                          magnitudes=magnitudes,
                                          binProbs=binProbs,
                                          binLengths=binLengths,
                                          trialProbs=trialProbs,
                                          stimOrder=stimOrder,
                                          trialProb=trialProb,
                                          trialOut=trialOut,
                                          trialMag=trialMag,
                                          itiDur=itiDur,
                                          isiDur=isiDur,
                                          fbDur=fbDur,
                                          highChosen=highChosen,
                                          selectedStim=selectedStim,
                                          sessionOnsets=sessionOnsets,
                                          sessionResponses=sessionResponses,
                                          stimAttrib=stimAttrib,
                                          payOut=payOut))
    taskInfo.__dict__.update({'sessionInfo': sessionInfo})
    return(taskInfo)
