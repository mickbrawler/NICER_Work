import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re
import scipy.stats as st

# This file's functions will produce the proper distribution for the NICER project

def eos_radii_lambdas(eos_name, N, m_sigma, r_sigma, name="test"):
    # Function that produces the possible masses, radii, lambdas for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    working_masses = []
    working_radii = []
    working_Lambdas = []
    Lambdas = []
    N_count = 0
    while N_count < N:
        try:
            m = np.random.normal(1.4, m_sigma, 1)[0]
            if m < 1.0: continue
            radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            rr = np.random.normal(radius, r_sigma, 1)[0]
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambdas = (2/3)*kk/(cc**5)
            working_Lambdas.append(Lambdas)
            working_masses.append(m)
            working_radii.append(rr)
            N_count += 1
        except RuntimeError:
            continue

    output = np.vstack((working_masses,working_radii,working_Lambdas)).T
    outputfile = "NICER_mock_data/mean_radii_lambdas/{}.txt".format(name)
    np.savetxt(outputfile, output, fmt="%f\t%f\t%f")

def plot_radii(datafile, name):
    # Function to plot eos' radii

        pl.clf()
        data = np.loadtxt(datafile)
        masses = data[:,0]
        radii = data[:,1]

        pl.rcParams.update({"font.size":18})
        pl.figure(figsize=(20,10))
        pl.scatter(masses,radii)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.title("Radii vs Masses")
        pl.savefig("NICER_mock_data/radii_plots/{}.png".format(name))

def plot_radii_heat(datafile, bins, name, save=True):
    # Function to plot eos' radii as heatmap

    pl.clf()
    data = np.loadtxt(datafile)
    masses = data[:,0]
    radii = data[:,1]

    heatmap, xedges, yedges = np.histogram2d(masses, radii, bins=bins) # creates a 2d grid with the intensities, and the coordinate values for each bin

    pl.imshow(heatmap.T, origin='lower') # Apparently imshow transposes the image and plots the origin in the upper left by default so this is accounted for
    pl.xlabel("Mass")
    pl.ylabel("Radius")
    pl.colorbar()

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(name))

def radii_gaussian_kde_plot(datafile, name, save=True):
    # Plot the kde of the eos' radii distribution

    pl.clf()
    data = np.loadtxt(datafile)
    m = data[:,0] 
    r = data[:,1]

    m_min, m_max = 1, 2.2 # 1.001069, 2.157369
    r_min, r_max = 9200, 13200 # 9242.634454, 13119.70321

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arraysgg
    positions = np.vstack([mm.ravel(), rr.ravel()])
    pairs = np.vstack([m, r])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, mm.shape)

    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(r_min, r_max)

    cfset = ax.pcolormesh(mm, rr, f, cmap='Blues')
    ax.set_xlabel('Mass')
    ax.set_ylabel('Radius')
    pl.scatter(m,r)

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(name))

def plot_lambdas(datafile, name):
    # Function to plot eos' lambdas

        pl.clf()
        data = np.loadtxt(datafile)
        masses = data[:,0]
        lambdas = data[:,2]

        pl.rcParams.update({"font.size":18})
        pl.figure(figsize=(20,10))
        pl.scatter(masses,lambdas)
        pl.xlabel("Mass")
        pl.ylabel("Lambdas")
        pl.title("Lambdas vs Masses")
        pl.savefig("NICER_mock_data/lambdas_plots/{}.png".format(name))

def plot_multiple_lambdas(datafile, name, N):
    # Function to plot eos' mean produced lambdas and multiple separate eos' lambda curves
    
    pl.clf()
    data = np.loadtxt(datafile)
    masses = data[:,0]
    Lambdas = data[:,2]
    pl.rcParams.update({"font.size":18})
    pl.figure(figsize=(20,10))
    pl.scatter(masses,Lambdas,color="brown")

    all_eos = lalsim.SimNeutronStarEOSNames
    GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2",
                "SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP",
                "SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1",
                "MS1B_PP","MS1_PP","BBB2","AP4","MPA1","MS1B","MS1",
                "SLY"]

    short_eos_list = ["APR4_EPP","SKOP","MPA1"]
    for eos_name in short_eos_list:
        eos = lalsim.SimNeutronStarEOSByName(eos_name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        m_min = 1.0
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        masses = np.linspace(m_min, max_mass, N)
        masses = masses[masses <= max_mass]

        working_masses = []
        working_Lambdas = []
        Lambdas = []
        for m in masses:
            try:
                radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
                cc = m*lal.MRSUN_SI/rr
                Lambdas = (2/3)*kk/(cc**5)
                working_Lambdas.append(Lambdas)
                working_masses.append(m)
            except RuntimeError:
                break
        pl.plot(masses,working_Lambdas,label=eos_name)
    pl.title("Lambdas vs Masses")
    pl.legend()
    pl.savefig("NICER_mock_data/multiple_lambdas_plots/{}.png".format(name))
