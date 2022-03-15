import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json
import scipy.stats as st
import seaborn as sns
import os
import glob

def global_max_dictionary(filename, outputfile, save=False):
    # Given parameter distribution dictionary, it produces a global maximum

    with open(filename, "r") as f:
        data = json.load(f)

    m_eos_val = {}

    max_ind = np.argmax(data["l"])

    max_p1 = data["p1"][max_ind]
    max_g1 = data["g1"][max_ind]
    max_g2 = data["g2"][max_ind]
    max_g3 = data["g3"][max_ind]
    max_l = data["l"][max_ind]
    m_eos_val.update({"eos":[max_p1,max_g1,max_g2,max_g3,max_l]})

    if save == True:
        with open(outputfile,"w") as f:
            json.dump(m_eos_val, f, indent=2, sort_keys=True)
    else: return([max_p1,max_g1,max_g2,max_g3,max_l])

def run_plot_m_eos_val_on_kde(datafile, name, N=1000):
    # Plot the heat and kde of the eos' radii distribution

    pl.clf()

    masses, radii = np.loadtxt(datafile, unpack=True) # Mass Radius distribution of mock data
    pairs = np.vstack([masses, radii])
    kernel = st.gaussian_kde(pairs)

    m_min, m_max = min(masses), max(masses) # 1.001069, 2.157369
    r_min, r_max = min(radii), max(radii) # 9242.634454, 13119.70321

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arraysgg
    positions = np.vstack([mm.ravel(), rr.ravel()])
    f = np.reshape(kernel(positions).T, mm.shape)

    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(r_min, r_max)

    ax.pcolormesh(mm, rr, f)
    ax.set_xlabel('Mass')
    ax.set_ylabel('Radius')

    Files = glob.glob("NICER_mock_data/MCMC_results/distributions/multiple_similar_distros/*.json")

    for File in Files:

        log_p1_SI, g1, g2, g3, _ = global_max_dictionary(File, "whatever", save=False)

        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        m_grid = np.linspace(m_min, max_mass, N)
        m_grid = m_grid[m_grid <= max_mass]

        working_masses = []
        working_radii = []
        for m in m_grid:
            try:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            except RuntimeError:
                continue
            except IndexError:
                continue
        pl.plot(working_masses,working_radii,color = "gray")
        
    pl.savefig("NICER_mock_data/MCMC_results/mass_radii_plots/mass_radii_{}.png".format(name))

def run_plot_parameter_distribution(name):
    # plots distributions of parameters found through MCMC

    pl.clf()

    APR4_EPP_p1 = 33.275399244401434
    APR4_EPP_g1 = 2.881652000854998
    APR4_EPP_g2 = 3.380399843127479
    APR4_EPP_g3 = 3.2762125079818984

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))

    Files = glob.glob("NICER_mock_data/MCMC_results/distributions/multiple_similar_distros/*.json")

    for File in Files:

        with open(File,"r") as f:
            data = json.load(f)

        sns.kdeplot(data["p1"],ax=axes[0,0],color="gray").set_title("Pressure")
        axes[0,0].axvline(APR4_EPP_p1)
        sns.kdeplot(data["g1"],ax=axes[0,1],color="gray").set_title("Gamma 1")
        axes[0,1].axvline(APR4_EPP_g1)
        sns.kdeplot(data["g2"],ax=axes[1,0],color="gray").set_title("Gamma 2")
        axes[1,0].axvline(APR4_EPP_g2)
        sns.kdeplot(data["g3"],ax=axes[1,1],color="gray").set_title("Gamma 3")
        axes[1,1].axvline(APR4_EPP_g3)

        pl.tight_layout()

    pl.savefig("NICER_mock_data/MCMC_results/parameter_plots/parameter_space_{}.png".format(name))

def combine_dictionaries(outputfile):
    # Combines data from multiple MCMC runs

    Files = glob.glob("NICER_mock_data/MCMC_results/distributions/multiple_similar_distros/*.json")


    p1 = []
    g1 = []
    g2 = []
    g3 = []
    l = []
    for File in Files:

        with open(File,"r") as f:
            data = json.load(f)

        p1 += data["p1"]
        g1 += data["g1"]
        g2 += data["g2"]
        g3 += data["g3"]
        l += data["l"]
    
    m_eos_val = {"p1":p1, "g1":g1, "g2":g2,"g3":g3, "l":l}

    with open(outputfile, "w") as f:
        json.dump(m_eos_val, f, indent=2, sort_keys=True)

