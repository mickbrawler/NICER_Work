from GWXtreme import eos_prior as ep
from multiprocessing import cpu_count, Pool
from scipy import interpolate
import scipy.stats as st
import lalsimulation as lalsim
import lal
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import json
import emcee
import math
import random
import argparse

def namedEoS_p_rho(EoS_Name):
    
    N = 1000
    min_log_pressure = 31.7
    max_log_pressure = 35.0

    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N+1)
    logp_grid = logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error

    density_grid = []
    for lp in logp_grid:

        eos = lalsim.SimNeutronStarEOSByName(EoS_Name)
        try:
            density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
        except RuntimeError: 
            continue # ran into runtime error at some point due to energydensityofpressure function

    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    pl.plot(np.log10(density_grid), logp_grid, label=EoS_Name)

    pl.xlim([17.1, 18.25])
    pl.xlabel(r'$log10(\frac{kg}{m^3})$')
    pl.ylabel(r'$log10(Pa)$')
    pl.legend()
    pl.savefig("plots/"+EoS_Name+"_p_rho.png", bbox_inches='tight')

def overlap_namedEoS_constraint_p_rho(EoS_Names):

    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    #File = "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0030_confidence.txt"
    File = "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_J0030_hierarchical_confidence.txt"
    #label = "X-ray obs"
    label = "Joint"
    #color = "#7570b3"
    color = "#000000"
    logp_grid, lower_bound, median, upper_bound = np.loadtxt(File).T

    ax1 = pl.gca()
    logp_grid = 10**logp_grid
    pl.xscale("log")
    pl.yscale("log")

    for EoS_Name in EoS_Names:

        density_grid = []
        safety_logp_grid = []
        eos = lalsim.SimNeutronStarEOSByName(EoS_Name)
        for lp in logp_grid:

            try:
                density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(lp, eos)/lal.C_SI**2)
                safety_logp_grid.append(lp)
            except RuntimeError: 
                continue # ran into runtime error at some point due to energydensityofpressure function

        #pl.plot(density_grid, safety_logp_grid, label="Theoretical Model", linestyle="dashed", linewidth=4)
        pl.plot(density_grid, safety_logp_grid, label=EoS_Name, linewidth=4)

    pl.plot(lower_bound, logp_grid, label=label, color=color)
    pl.plot(upper_bound, logp_grid, color=color)
    ax1.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color=color, alpha=0.45)

    pl.vlines(x=2.3*10**17,ymin=min(logp_grid),ymax=max(logp_grid),color="gray")
    pl.text(1.75*10**17,10**31.75,"Nuclear Density",fontsize=16,color="gray")
    pl.text(10**17.75,10**31.75,"Super-Nuclear Density",fontsize=16,color="black")
    pl.xlim([10**16.99, 10**18.25])
    pl.ylim([min(logp_grid), max(logp_grid)])
    pl.xlabel('Density')
    pl.ylabel('Pressure')
    pl.legend()
    #pl.savefig("plots/overlap_APR4EPP_EM_constraint_p_rho.png", bbox_inches='tight')
    pl.savefig("plots/overlap_APR4EPP_Joint_constraint_p_rho.png", bbox_inches='tight')

