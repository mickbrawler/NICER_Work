import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re
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
