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
import scipy.stats as st

# This file's functions will produce the proper mass radius distribution for the NICER project

def namedEoS_lines(EoS_Name):
    
    ### pressure density
    N = 1000
    min_log_pressure = 31.7
    max_log_pressure = 35.0

    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N+1)
    logp_grid = logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error

    eos = lalsim.SimNeutronStarEOSByName(EoS_Name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    density_grid = []
    for lp in logp_grid:

        try:
            density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
        except RuntimeError: 
            continue # ran into runtime error at some point due to energydensityofpressure function

    pl.clf()
    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    pl.plot(np.log10(density_grid), logp_grid, label=EoS_Name)

    pl.xlim([min(np.log10(density_grid)),max(np.log10(density_grid))])
    pl.xlabel(r'$log10(\frac{kg}{m^3})$')
    pl.ylabel(r'$log10(Pa)$')
    pl.legend()
    pl.savefig("plots/AstroClub2023/"+EoS_Name+"_p_rho.png", bbox_inches='tight')

    ### radius mass
    m_min = 1.0
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]

    working_masses = []
    working_radii = []
    working_compactnesses = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            working_masses.append(m)
            working_compactnesses.append(cc)
            working_radii.append(rr)
        except RuntimeError:
            break

    working_radii = np.array(working_radii) / 1000

    pl.clf()
    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    pl.plot(working_masses, working_radii, label=EoS_Name)

    pl.xlim([min(working_masses),max(working_masses)])
    pl.xlabel("Mass (M$\odot$)")
    pl.ylabel("Radius (km)")
    pl.legend()
    pl.savefig("plots/AstroClub2023/"+EoS_Name+"_r_m.png", bbox_inches='tight')

    ### compactness mass
    pl.clf()
    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    pl.plot(working_masses, working_compactnesses, label=EoS_Name)

    pl.xlim([min(working_masses),max(working_masses)])
    pl.xlabel("Mass (M$\odot$)")
    pl.ylabel("Compactness")
    pl.legend()
    pl.savefig("plots/AstroClub2023/"+EoS_Name+"_c_m.png", bbox_inches='tight')

    ### chirp-mass ~tidal-deformbability

    #q[0.7,1.0]
    #lambda[0,2000]
    # Not as simple to do as radius mass data. lalsim has no function for it.

def plot_scatter_AND_gaussian_kde_scatter(datafile, name, labels, turn_to_km=False):
    # Plot the scatter and kde/scatter of a data file posterior
    #labels = ["Mass (M$\odot$)","Radius (km)",]
    #labels = ["Mass (M$\odot$)","Compactness",]
    #labels = ["$\\tilde{\\Lambda}$","$q$"]

    ### scatter
    pl.clf()
    data = np.loadtxt(datafile)
    x = data[:,0]
    if turn_to_km: y = data[:,1] / 1000
    else: y = data[:,1]
    
    pl.rcParams.update({"font.size":20})
    pl.figure(figsize=(12,12))
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    pl.scatter(x,y,s=3)

    pl.xlabel(labels[0])
    pl.ylabel(labels[1])
    pl.savefig("plots/AstroClub2023/scatter_{}.png".format(name), bbox_inches='tight')

    ### kde/scatter
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Perform the kernel density estimate
    xx, yy = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j] # two 2d arrays
    positions = np.vstack([xx.ravel(), yy.ravel()])
    pairs = np.vstack([x, y])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, xx.shape)

    pl.clf()
    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    pl.rcParams.update({"font.size":20})
    pl.figure(figsize=(12,12))
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    ax.pcolormesh(xx, yy, f)
    #pl.scatter(x,y,s=1,color="black")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    pl.savefig("plots/AstroClub2023/kde_scatter_{}.png".format(name), bbox_inches='tight')

def overlap_namedEoS_constraint_p_rho(EoS_Names):

    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)

    File = "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_J0030_hierarchical_confidence.txt"
    label = "Joint"
    color = "#000000"
    logp_grid, lower_bound, median, upper_bound = np.loadtxt(File).T

    ax1 = pl.gca()
    pl.plot(np.log10(lower_bound), logp_grid, label=label, color=color)
    pl.plot(np.log10(upper_bound), logp_grid, color=color)
    ax1.fill_betweenx(logp_grid, np.log10(lower_bound), x2=np.log10(upper_bound), color=color, alpha=0.45)

    for EoS_Name in EoS_Names:

        density_grid = []
        safety_logp_grid = []
        eos = lalsim.SimNeutronStarEOSByName(EoS_Name)
        for lp in logp_grid:

            try:
                density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
                safety_logp_grid.append(lp)
            except RuntimeError: 
                continue # ran into runtime error at some point due to energydensityofpressure function

        pl.plot(np.log10(density_grid), safety_logp_grid, label=EoS_Name, linewidth=4)

    pl.xlim([16.99, 18.25])
    pl.xlabel(r'$log10(\frac{kg}{m^3})$')
    pl.ylabel(r'$log10(Pa)$')
    pl.legend()
    pl.savefig("plots/AstroClub2023/overlap_proposedEoS_Joint_constraint_p_rho.png", bbox_inches='tight')

