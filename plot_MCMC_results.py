import numpy as np
import pylab as pl
import seaborn as sns
import json

GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
            "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
            "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP","BBB2","AP4",
            "MPA1","MS1B","MS1","SLY"]

def plotter(filename, eos, eos_file_name):
    # plots distributions of parameters found through MCMC
    #file names must include directory and type.

    with open(filename,"r") as f:
        data = json.load(f)

    r_val_max = np.argmax(data[eos]["r2"])

    p1_val_max = data[eos]["p1"][r_val_max]
    g1_val_max = data[eos]["g1"][r_val_max]
    g2_val_max = data[eos]["g2"][r_val_max]
    g3_val_max = data[eos]["g3"][r_val_max]

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))
    fig.suptitle("{}: [{},{},{},{}]".format(eos,p1_val_max,g1_val_max,g2_val_max,g3_val_max),fontsize=11)
    
    sns.kdeplot(data[eos]["p1"],ax=axes[0,0]).set_title("Pressure: {}".format(p1_val_max))
    axes[0,0].axvline(data[eos]["p1"][r_val_max])
    sns.kdeplot(data[eos]["g1"],ax=axes[0,1]).set_title("Gamma 1: {}".format(g1_val_max))
    axes[0,1].axvline(data[eos]["g1"][r_val_max])
    sns.kdeplot(data[eos]["g2"],ax=axes[1,0]).set_title("Gamma 2: {}".format(g2_val_max))
    axes[1,0].axvline(data[eos]["g2"][r_val_max])
    sns.kdeplot(data[eos]["g3"],ax=axes[1,1]).set_title("Gamma 3: {}".format(g3_val_max))
    axes[1,1].axvline(data[eos]["g3"][r_val_max])

    pl.tight_layout()
    pl.savefig(eos_file_name)

def plotter_runner(eos_list, directory, filename):

    for eos in eos_list:

        plotter(filename,eos,"{}{}_Refined_parameter_density.png".format(directory,eos))
