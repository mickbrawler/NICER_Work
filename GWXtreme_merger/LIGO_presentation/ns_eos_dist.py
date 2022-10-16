import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re
import scipy.stats as st

# This file's functions will produce the proper mass radius distribution for the NICER project

def mr_posterior_trimmer(filename, size, label):
    # Function that trims large real NICER m-r posterior to lower size
    
    data = np.loadtxt(filename)
    choices_index = np.random.randint(len(data), size=size)
    trimmed_data = data[choices_index,:]
    radii = trimmed_data[:,0] * 1000
    masses = trimmed_data[:,1]

    trimmed_vals = np.vstack((masses, radii)).T
    outputfile = "new_data/mass_radii_{}.txt".format(label)
    np.savetxt(outputfile, trimmed_vals, fmt="%f\t%f")

def eos_radii_posterior(eos_name, N, m_sigma, r_sigma, label):
    # Function that produces the possible masses and radii for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    working_masses = []
    working_radii = []
    N_count = 0
    while N_count < N:
        try:
            m = np.random.normal(1.4, m_sigma, 1)[0] # mean mass of 1.4, standard deviation of m_sigma
            if m < 1.0: continue # mass can't be less than 1 solar mass
            radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam) # actual radius for m given eos
            rr = np.random.normal(radius, r_sigma, 1)[0] # tampered radius to simulate weak signal
            working_masses.append(m)
            working_radii.append(rr)
            N_count += 1
        except RuntimeError: # ??? Error may have resulted from radius calculation (bad m)
            continue

    output = np.vstack((working_masses,working_radii)).T
    ###outputfile = "NICER_mock_data/mass_radii_posterior/mass_radii_{}.txt".format(label) # label="APR4_EPP_N????"
    outputfile = "new_data/mass_radii_{}.txt".format(label)
    np.savetxt(outputfile, output, fmt="%f\t%f")

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
        pl.title("Radii vs Masses")
        ###pl.savefig("NICER_mock_data/radii_plots/{}.png".format(label)) # label="APR4_EPP_N????"
        pl.savefig("plots/scatter_{}.png".format(label)) # label="APR4_EPP_N????"

def plot_radii_gaussian_kde(datafile, label, EoS=False):
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
    if type(EoS) == str:
        mass, radii = np.loadtxt(EoS).T
        radii = radii / 1000
        pl.plot(mass, radii, color="red")
    pl.title("Mock Radius-Mass Distribution")
    pl.savefig("plots/{}.png".format(label), bbox_inches='tight') # label="APR4_EPP_m(m_sigma)_r(r_sigma)_kde_mesh_scatter"

