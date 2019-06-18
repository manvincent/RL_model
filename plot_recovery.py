# Load in data 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/analysis/modelling')

homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/med_lat_OFC/analysis/modelling/param_recov'
os.chdir(homeDir) 

# Across experiment plots
# Import data
recovCorrDF = pd.read_csv(homeDir + os.sep + 'paramRecov_results.csv')
# Only show up to 120 trials
recovCorrDF = recovCorrDF.loc[:11]
# Plot group results
fig3 = plt.figure() 
fig3.suptitle("Recoverability Across Samples")
fig3.text(0.5, 0.04, '# Trials', ha='center')
fig3.text(0.04, 0.5, "Pearson's r", va='center', rotation='vertical')  
ax3 = fig3.add_subplot(1,1,1)
sns.regplot(recovCorrDF.sample_n, 
            recovCorrDF.alpha, 
            order=1,
            logistic=True,
            ci=False,            
            ax=ax3); 
sns.regplot(recovCorrDF.sample_n, 
            recovCorrDF.beta, 
            order=1,
            logistic=True,
            ci=False,
            color='red',
            ax=ax3); 
ax3.axvline(70,color='red')
ax3.axvline(110,color='red',linestyle="--")
ax3.set_xlabel('')
ax3.set_ylabel('')
 
# Per experiment / trialNo plots

# Initialize plots
fig1  = plt.figure()
fig1.subplots_adjust(hspace=0.5)
fig1.suptitle("Parameter Recovery: Learning Rate")
fig1.text(0.5, 0.04, 'Generative parameters', ha='center')
fig1.text(0.04, 0.5, 'Recovered parameters', va='center', rotation='vertical')  

fig2  = plt.figure()
fig2.subplots_adjust(hspace=0.5)
fig2.suptitle("Parameter Recovery: Inverse Temperature")
fig2.text(0.5, 0.04, 'Generative parameters', ha='center')
fig2.text(0.04, 0.5, 'Recovered parameters', va='center', rotation='vertical')  
# Loop through different 'experiments' of varying trial numbers
for sampleIdx, numTrials in enumerate(np.arange(1,16)*10):
    # Load data
    sampleDF = pd.read_csv(homeDir + os.sep + 'Recover' + os.sep + 'n_' + str(numTrials) + '_paramRecov.csv')
    # Plot learning rate recoverability
    ax1 = fig1.add_subplot(1, len(np.arange(1,16)*10), sampleIdx+1)
    sns.regplot(sampleDF.gen_alpha,
                sampleDF.recov_alpha,
                scatter=True,
                order=1,
                ci=False,
                color='blue',
                scatter_kws={'alpha':0.3},
                ax=ax1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # Plot inverse temperature recoverability 
    ax2 = fig2.add_subplot(1, len(np.arange(1,16)*10), sampleIdx+1)
    sns.regplot(sampleDF.gen_beta,
                sampleDF.recov_beta,
                scatter=True,
                order=1,
                ci=False,
                color='red',
                scatter_kws={'alpha':0.3},
                ax=ax2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
# Save figures
fig1.savefig(initDict.outDir + os.sep + 'qA_recov.svg', format='svg')
fig2.savefig(initDict.outDir + os.sep + 'smB_recov.svg',format='svg')    
fig3.savefig(initDict.homeDir + os.sep + 'paramRecov_results.svg', format='svg')    