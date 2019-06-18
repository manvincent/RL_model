#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:01:26 2019

@author: vman
"""
# General functions to create onsets given input pandas dataframes 
# and fmri software style

import numpy as np
import pandas as pd
import os
homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/analysis/data'

def createOnsets(subID, timeDF, expDF, compVarDF, cueVar, outVar, style = 'FSL'):
    # Create output directories
    if style == 'FSL':
        outDir = f'{homeDir}/fsl_onsets'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        if not os.path.exists(f'{outDir}/cue_onsets'):
            os.makedirs(f'{outDir}/cue_onsets')
        if not os.path.exists(f'{outDir}/outcome_onsets'):
            os.makedirs(f'{outDir}/outcome_onsets')
            
    # Cue-timed onsets 
    cueOnset = np.round(timeDF.tStim, 2)
    cueDur = np.round(expDF.RT, 2)
    # Create intercept cue onset
    cueDF = pd.DataFrame(columns=['onset','duration','param'])
    cueDF.onset = cueOnset
    cueDF.duration = cueDur
    cueDF.param = np.ones(len(cueOnset)).astype(int)
    cueDF = cueDF.dropna(how = 'any')
    cueDF.to_csv(f'{outDir}/cue_onsets/s{subID}_cue_onsets.txt',
                 sep=' ',
                 index = False,
                 header = False)
    # Create parametrically modulated onset files 
    for varIdx, var in enumerate(cueVar):
        param = []
        param = compVarDF[cueVar[varIdx]]
        # Mean center the parametric modulator 
        param_mc = np.round(param - np.nanmean(param), 2)
        if style == 'FSL':            
            # Initialize onset dataframes
            onsetDF = pd.DataFrame(columns=['onset','duration','param'])
            onsetDF.onset = cueOnset
            onsetDF.duration = cueDur
            onsetDF.param = param_mc
            onsetDF = onsetDF.dropna(how = 'any')
            onsetDF.to_csv(f'{outDir}/cue_onsets/s{subID}_{var}_onsets.txt',
                           sep=' ',
                           index = False, 
                           header = False)
    # Cue-timed onsets 
    outOnset = np.round(timeDF.tOut, 2)
    outDur = 1.75
    # Create intercept cue onset
    outDF = pd.DataFrame(columns=['onset','duration','param'])
    outDF.onset = outOnset
    outDF.duration = outDur
    outDF.param = np.ones(len(outOnset)).astype(int) 
    outDF = outDF.dropna(how = 'any')
    outDF.to_csv(f'{outDir}/outcome_onsets/s{subID}_outcome_onsets.txt',
                 sep=' ',
                 index = False,
                 header = False)
    # Create parametrically modulated onset files 
    for varIdx, var in enumerate(outVar):
        param = []
        param = compVarDF[outVar[varIdx]]
        # Mean center the parametric modulator 
        param_mc = np.round(param - np.nanmean(param), 2)
        if style == 'FSL':            
            # Initialize onset dataframes
            onsetDF = pd.DataFrame(columns=['onset','duration','param'])
            onsetDF.onset = outOnset
            onsetDF.duration = outDur
            onsetDF.param = param_mc
            onsetDF = onsetDF.dropna(how = 'any')
            onsetDF.to_csv(f'{outDir}/outcome_onsets/s{subID}_{var}_onsets.txt',
                           sep=' ',
                           index = False, 
                           header = False)
    return


