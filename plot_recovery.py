# Load in data 
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling')
homeDir = '/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling/param_recovery'
os.chdir(homeDir) 

# Across experiment plots
# Import data
numMaxDays = 6
sessPerDays = 5
recovCorrDF = pd.DataFrame(columns=['alpha', 'beta', 'numDays','lo_alpha','hi_alpha','lo_beta','hi_beta'])
for sampleIdx, numSessions in enumerate(np.arange(1,numMaxDays+1) * sessPerDays): 
    sampleDF = pd.read_csv(f'{homeDir}/Recover/sessions_{numSessions}_paramRecov.csv')
    corr_alpha, _ = stats.pearsonr(sampleDF.gen_alpha, sampleDF.recov_alpha)
    corr_beta, _ = stats.pearsonr(sampleDF.gen_beta, sampleDF.recov_beta)
    # Compute confidence intervals
    z = stats.norm.ppf(1-0.05/2)
    se = 1/np.sqrt(sampleDF.gen_alpha.size-3)
    
    z_alpha = np.arctanh(corr_alpha)
    z_beta = np.arctanh(corr_beta)
    lo_z_alpha, hi_z_alpha = z_alpha-z*se, z_alpha+z*se
    lo_z_beta, hi_z_beta = z_beta-z*se, z_beta+z*se
    
    lo_alpha, hi_alpha = np.tanh((lo_z_alpha, hi_z_alpha))
    lo_beta, hi_beta = np.tanh((lo_z_beta, hi_z_beta))
    # Append to dataframe
    numDays = numSessions/sessPerDays
    recovCorrDF = recovCorrDF.append({'alpha':corr_alpha,
                                      'beta':corr_beta,
                                      'numDays':numDays,
                                      'lo_alpha':lo_alpha,
                                      'hi_alpha':hi_alpha,
                                      'lo_beta':lo_beta,
                                      'hi_beta':hi_beta}, ignore_index=True)
    
# Plot group results
fig3 = plt.figure() 
fig3.suptitle("Recoverability Across Days")
fig3.text(0.5, 0.04, '# Trials', ha='center')
fig3.text(0.04, 0.5, "Pearson's r", va='center', rotation='vertical')  
ax3 = fig3.add_subplot(1,1,1)

sns.pointplot(recovCorrDF.numDays, 
            recovCorrDF.alpha, 
            color='blue',
            ax=ax3); 
sns.lineplot(recovCorrDF.numDays-1,
             np.array(recovCorrDF.lo_alpha),
             color='blue',
             alpha=0.3,
             ax=ax3)
sns.lineplot(recovCorrDF.numDays-1,
             np.array(recovCorrDF.hi_alpha),
             color='blue',
             alpha=0.3,
             ax=ax3)
sns.pointplot(recovCorrDF.numDays, 
            recovCorrDF.beta, 
            color='red',
            ax=ax3); 
sns.lineplot(recovCorrDF.numDays-1,
             np.array(recovCorrDF.lo_beta),
             color='red',
             alpha=0.3,
             ax=ax3)              
sns.lineplot(recovCorrDF.numDays-1,
             np.array(recovCorrDF.hi_beta),
             color='red',
             alpha=0.3,
             ax=ax3)              

#ax3.axvline(70,color='red')
#ax3.axvline(110,color='red',linestyle="--")
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
for sampleIdx, numSessions in enumerate(np.arange(1,numMaxDays+1) * sessPerDays): 
    sampleDF = pd.read_csv(f'{homeDir}/Recover/sessions_{numSessions}_paramRecov.csv')
    # Plot learning rate recoverability
    ax1 = fig1.add_subplot(1, len(np.arange(1,numMaxDays+1) * sessPerDays), sampleIdx+1)
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
    ax2 = fig2.add_subplot(1, len(np.arange(1,numMaxDays+1) * sessPerDays), sampleIdx+1)
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
fig1.savefig(f'{homeDir}/qA_recov.svg', format='svg')
fig2.savefig(f'{homeDir}/smB_recov.svg',format='svg')    
fig3.savefig(f'{homeDir}/paramRecov_results.svg', format='svg')    