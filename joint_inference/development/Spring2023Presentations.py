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

def mr_posterior_trimmer(filename, size, outputfile):
    # Function that trims large real NICER m-r posterior to lower size
    
    data = np.loadtxt(filename)
    choices_index = np.random.randint(len(data), size=size)
    trimmed_data = data[choices_index,:]
    radii = trimmed_data[:,0] * 1000
    masses = trimmed_data[:,1]

    trimmed_vals = np.vstack((masses, radii)).T
    np.savetxt(outputfile, trimmed_vals, fmt="%f\t%f")

def plot_radii_scatter(datafile, label):
    # Function to plot scatter of eos' radii

        pl.clf()
        data = np.loadtxt(datafile)
        masses = data[:,0]
        ###radii = data[:,1]
        radii = data[:,1] / 1000

        pl.rcParams.update({"font.size":18})
        pl.figure(figsize=(20,10))
        pl.scatter(masses,radii,s=5)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.savefig("plots/AstroClub2023/{}.png".format(label))

def plot_radii_gaussian_kde(datafile, label, save=True):
    # Plot the kde of the eos' radii distribution

    pl.clf()
    data = np.loadtxt(datafile)
    m = data[:,0] 
    r = data[:,1] / 1000

#    m_min, m_max = 1, 2.2 # 1.001069, 2.157369
#    r_min, r_max = 9200, 13200 # 9242.634454, 13119.70321

    m_min, m_max = min(m), max(m) # 1.001069, 2.157369
    r_min, r_max = min(r), max(r) # 9242.634454, 13119.70321

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arrays
    positions = np.vstack([mm.ravel(), rr.ravel()])
    pairs = np.vstack([m, r])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, mm.shape)

    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(r_min, r_max)

    ax.pcolormesh(mm, rr, f)
    ax.set_xlabel('Mass')
    ax.set_ylabel('Radius (km)')
    pl.scatter(m,r,s=1,color="black")
    pl.title("Mass-Radius Distribution")

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(label), bbox_inches='tight') # label="APR4_EPP_m(m_sigma)_r(r_sigma)_kde_mesh_scatter"

def plot_radii_kde_heat(datafile, bins, label, save=True, N=1000):
    # Plot the heat and kde of the eos' radii distribution

    pl.clf()
    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(20,15))
    pl.rcParams["font.size"] = "16"

    data = np.loadtxt(datafile)
    masses = data[:,0]
    radii = data[:,1]

    m_min, m_max = 1, 2.2 # 1.001069, 2.157369
    r_min, r_max = 9200, 13200 # 9242.634454, 13119.70321

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arraysgg
    positions = np.vstack([mm.ravel(), rr.ravel()])
    pairs = np.vstack([masses, radii])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, mm.shape)

    ax1.set_xlim(m_min, m_max)
    ax1.set_ylim(r_min, r_max)

    cfset = ax1.pcolormesh(mm, rr, f)
    ax1.scatter(masses,radii,s=5,color="black")

    short_eos_list = ["APR4_EPP","SKOP","MPA1"]
    for eos_name in short_eos_list:
        eos = lalsim.SimNeutronStarEOSByName(eos_name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        m_min = 1.0
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        mm = np.linspace(m_min, max_mass, N)
        mm = mm[mm <= max_mass]

        working_masses = []
        working_radii = []
        for m in mm:
            try:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            except RuntimeError:
                break
        ax1.plot(working_masses,working_radii,label=eos_name)
    ax1.legend()

    heatmap, xedges, yedges = np.histogram2d(masses, radii, bins=bins) # creates a 2d grid with the intensities, and the coordinate values for each bin
    
    ax2.imshow(heatmap.T, origin="lower") # Apparently imshow transposes the image and plots the origin in the upper left by default so this is accounted for

    pl.suptitle("APR4_EPP: Mass & Radius Distribution")
    fig.text(0.5, 0.01, "Mass", ha="center")
    fig.text(0.01, 0.5, "Radius", va="center", rotation="vertical")

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(label)) #label="APR4_EPP_m(m_sigma)_r(r_sigma)_sub"

def eos_mr_pd_curves(eos_name, N=1000):
    # Function that produces m-r curve and p-d curve plots for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    m_min = 1.0
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]

    working_masses = []
    working_radii = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            working_masses.append(m)
            working_radii.append(rr)
        except RuntimeError:
            break

    working_radii = np.array(working_radii) / 1000

    pl.clf()
    pl.plot(working_masses, working_radii)
    pl.xlabel("Mass")
    pl.ylabel("Radius (km)")
    pl.title("Mass-Radius Curve")
    pl.savefig("emcee_files/plots/{}_MR_Curve.png".format(eos_name), bbox_inches='tight')

    min_log_pressure = 32.0
    max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N)

    density_grid = []
    for lp in logp_grid:

        density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)

    pl.clf()
    ax = pl.gca()
    ax.set_xscale("log")
    pl.plot(density_grid, logp_grid)
    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.title("Pressure vs Density Curve")
    pl.savefig("emcee_files/plots/{}_PD_Curve.png".format(eos_name), bbox_inches='tight')

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

    pl.plot(np.log10(density_grid), logp_grid, label="EoS_Name")

    pl.xlim([17.1, 18.25])
    pl.xlabel(r'$log10(\frac{kg}{m^3})$')
    pl.ylabel(r'$log10(Pa)$')
    pl.legend()
    pl.savefig("plots/AstroClub2023/"+EoS_Name+"_p_rho.png", bbox_inches='tight')

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

        #pl.plot(density_grid, safety_logp_grid, label="Theoretical Model", linestyle="dashed", linewidth=4)
        pl.plot(np.log10(density_grid), safety_logp_grid, label=EoS_Name, linewidth=4)

    pl.xlim([16.99, 18.25])
    pl.xlabel(r'$log10(\frac{kg}{m^3})$')
    pl.ylabel(r'$log10(Pa)$')
    pl.legend()
    #pl.savefig("plots/overlap_APR4EPP_EM_constraint_p_rho.png", bbox_inches='tight')
    pl.savefig("plots/AstroClub2023/overlap_APR4EPP_Joint_constraint_p_rho.png", bbox_inches='tight')
