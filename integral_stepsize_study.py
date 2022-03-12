import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json
import scipy.stats as st
import seaborn as sns

def global_max_dictionary(filename1, filename2, outputfile, save=False):
# Given 2 parameter distribution dictionary, it produces the global maximum

    with open(filename1, "r") as f:
        data1 = json.load(f)

    with open(filename2, "r") as f:
        data2 = json.load(f)

    m_eos_val = {}

    max_ind_1 = np.argmax(data1["l"])
    max_p1_1 = data1["p1"][max_ind_1]
    max_g1_1 = data1["g1"][max_ind_1]
    max_g2_1 = data1["g2"][max_ind_1]
    max_g3_1 = data1["g3"][max_ind_1]
    max_l_1 = data1["l"][max_ind_1]
    m_eos_val.update({"eos1":[max_p1_1,max_g1_1,max_g2_1,max_g3_1,max_l_1]})

    max_ind_2 = np.argmax(data2["l"])
    max_p1_2 = data2["p1"][max_ind_2]
    max_g1_2 = data2["g1"][max_ind_2]
    max_g2_2 = data2["g2"][max_ind_2]
    max_g3_2 = data2["g3"][max_ind_2]
    max_l_2 = data2["l"][max_ind_2]
    m_eos_val.update({"eos2":[max_p1_2,max_g1_2,max_g2_2,max_g3_2,max_l_2]})

    if save == True:
        with open(outputfile,"w") as f:
            json.dump(m_eos_val, f, indent=2, sort_keys=True)
    else: return([[max_p1_1,max_g1_1,max_g2_1,max_g3_1,max_l_1],[max_p1_2,max_g1_2,max_g2_2,max_g3_2,max_l_2]])

def plot_m_eos_val_on_kde(datafile, MCMC_distribution_file1, MCMC_distribution_file2, name, N=1000):
    # Plot the heat and kde of the eos' radii distribution

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

    parameters_list = global_max_dictionary(MCMC_distribution_file1, MCMC_distribution_file2, "whatever", save=False)

    count = 0
    label_list = ["N=1000","N=10,000"]
    for parameters in parameters_list:

        log_p1_SI, g1, g2, g3, _ = parameters
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
        pl.plot(working_masses,working_radii,label=label_list[count])
        count += 1

    pl.legend()
    pl.savefig("NICER_mock_data/MCMC_results/mass_radii_plots/mass_radii_{}.png".format(name))

def plot_parameter_distribution(filename1, filename2, name):
    # plots distributions of parameters found through MCMC

    with open(filename1,"r") as f:
        data1 = json.load(f)

    with open(filename2,"r") as f:
        data2 = json.load(f)

    APR4_EPP_p1 = 33.275399244401434
    APR4_EPP_g1 = 2.881652000854998
    APR4_EPP_g2 = 3.380399843127479
    APR4_EPP_g3 = 3.2762125079818984

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))
    
    sns.kdeplot(data1["p1"],ax=axes[0,0]).set_title("Pressure")
    sns.kdeplot(data2["p1"],ax=axes[0,0]).set_title("Pressure")
    axes[0,0].axvline(APR4_EPP_p1)
    sns.kdeplot(data1["g1"],ax=axes[0,1]).set_title("Gamma 1")
    sns.kdeplot(data2["g1"],ax=axes[0,1]).set_title("Gamma 1")
    axes[0,1].axvline(APR4_EPP_g1)
    sns.kdeplot(data1["g2"],ax=axes[1,0]).set_title("Gamma 2")
    sns.kdeplot(data2["g2"],ax=axes[1,0]).set_title("Gamma 2")
    axes[1,0].axvline(APR4_EPP_g2)
    sns.kdeplot(data1["g3"],ax=axes[1,1]).set_title("Gamma 3")
    sns.kdeplot(data2["g3"],ax=axes[1,1]).set_title("Gamma 3")
    axes[1,1].axvline(APR4_EPP_g3)

    pl.tight_layout()
    pl.legend(labels=["N=1,000","N=10,000"])
    pl.savefig("NICER_mock_data/MCMC_results/parameter_plots/parameter_space_{}.png".format(name))

