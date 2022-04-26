import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re
import scipy.stats as st

# This file's functions will produce the proper mass radius distribution for the NICER project

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
    outputfile = "NICER_mock_data/mass_radii_posterior/mass_radii_{}.txt".format(label) # label="APR4_EPP_N????"
    np.savetxt(outputfile, output, fmt="%f\t%f")

def varying_std_posterior(eos_name, N):
    # Producing m-r posteriors of increasing m and r standard deviation
    
    # These sigmas may need to be changed (to more equal sigmas for both m and r)
    m_sigmas = [.2,.4,.6,.8,1]
    r_sigmas = [250,500,750,1000,1250]
    for sigma in range(len(m_sigmas)):
        eos_radii_posterior(eos_name, 1000, m_sigmas[sigma], r_sigmas[sigma], "{}_N{}_m{}_r{}".format(eos_name,N,m_sigmas[sigma],r_sigmas[sigma]))

def plot_radii_scatter(datafile, label):
    # Function to plot scatter of eos' radii

        pl.clf()
        data = np.loadtxt(datafile)
        masses = data[:,0]
        radii = data[:,1]

        pl.rcParams.update({"font.size":18})
        pl.figure(figsize=(20,10))
        pl.scatter(masses,radii,s=5)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.title("Radii vs Masses")
        pl.savefig("NICER_mock_data/radii_plots/{}.png".format(label)) # label="APR4_EPP_N????"

def plot_radii_gaussian_kde(datafile, label, save=True):
    # Plot the kde of the eos' radii distribution

    pl.clf()
    data = np.loadtxt(datafile)
    m = data[:,0] 
    r = data[:,1]

    m_min, m_max = 1, 2.2 # 1.001069, 2.157369
    r_min, r_max = 9200, 13200 # 9242.634454, 13119.70321

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
    ax.set_ylabel('Radius')
    pl.scatter(m,r,s=5,color="black")

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(label)) # label="APR4_EPP_m(m_sigma)_r(r_sigma)_kde_mesh_scatter"

def plot_radii_heat(datafile, bins, label, save=True):
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

    if save: pl.savefig("NICER_mock_data/radii_heat_plots/{}.png".format(label)) #label="APR4_EPP_m(m_sigma)_r(r_sigma)_bin(bins)"

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

def density_along_curve(datafile, N=1000):
    # Finds integral of eos curves over the kde of a mass-radius posterior sample

    masses, radii = np.loadtxt(datafile, unpack=True)
    pairs = np.vstack([masses, radii])
    kernel = st.gaussian_kde(pairs)

    short_eos_list = ["APR4_EPP","SKOP","MPA1"]
    sum_list = []
    for eos_name in short_eos_list:
        eos = lalsim.SimNeutronStarEOSByName(eos_name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        m_min = np.min(masses)
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
                break
        sum_list.append(np.sum(kernel(np.vstack([working_masses, working_radii]))*np.diff(working_masses)[0]))
    return(sum_list)

def likelihood_piecewise(datafile, params, N=1000):
    # Finds integral of eos curve over the kde of a mass-radius posterior sample

    # Bottom 3 lines should be separated into init() of a class in the future
    masses, radii = np.loadtxt(datafile, unpack=True)
    pairs = np.vstack([masses, radii])
    kernel = st.gaussian_kde(pairs)

    log_p1_SI, g1, g2, g3 = params
    eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_min = np.min(masses)
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
            break
    integral = np.sum(kernel(np.vstack([working_masses, working_radii]))*np.diff(working_masses)[0])
    return(integral)

def error_barplot(datafile, label):
    # Barplot of error of low N eos curve integral values to that of high ones

    eos_list = ["BHF_BBB2","KDE0V","SKOP","RS","APR4_EPP","SKI6","MPA1","AP4"]
    Ns = [100,1000,100000]

    N1_integral = []
    N2_integral = []
    true_integral = []
    for eos_name in eos_list:
        for N in Ns:
            integral = likelihood(datafile, eos_name, N=N)
            if Ns.index(N)==0: N1_integral.append(integral)
            elif Ns.index(N)==1: N2_integral.append(integral)
            else: true_integral.append(integral)
    
    N1_integral = np.array(N1_integral)
    N2_integral = np.array(N2_integral)
    true_integral = np.array(true_integral)
    N1 = abs(true_integral - N1_integral) / true_integral
    N2 = abs(true_integral - N2_integral) / true_integral

    x = np.arange(len(eos_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = pl.subplots()
    ax.bar(x - width/2, N1, width, label="N_100")
    ax.bar(x + width/2, N2, width, label="N_1000")

    ax.set_xlabel('EOS Curve Sizes')
    ax.set_ylabel('Error')
    ax.set_xticks(x)
    ax.set_xticklabels(eos_list)
    ax.legend()
    pl.savefig("NICER_mock_data/error_plots/{}.png".format(label)) # label="APR4_EPP_N????"

# altered
#p1 [33.275399244401434-33.55080882234776]
#g1 [2.6168480380843007-3.2142052715754117]
#g2 [2.8393440753622485-3.380399843127479]
#g3 [2.5282721765465537-3.2762125079818984]

def plot_random_eos_curves(label, n_pp, n_n):
    # Plot n random eos piecewise parametrized curves of mass-radii, and n number of names eos mass-radii
    
    pl.clf()

    N=1000
    p1_incr, g1_incr, g2_incr, g3_incr = .4575, .927, 1.1595, .9285
    log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 

#    p1_incr, g1_incr, g2_incr, g3_incr = 0.13770478897,0.29867861674, 0.27052788388, 0.37397016571
#    log_p1_SI,g1,g2,g3 = 33.4131040334, 2.91552665482, 3.10987195924, 2.90224234226

    n_count = 0
    while n_count < n_pp:
        try:
            log_p1_SI = ((log_p1_SI - (.25 * p1_incr)) + ((2 * (.25 * p1_incr)) * np.random.random()))
            g1 = ((g1 - (.25 * g1_incr)) + ((2 * (.25 * g1_incr)) * np.random.random()))
            g2 = ((g2 - (.25 * g2_incr)) + ((2 * (.25 * g2_incr)) * np.random.random()))
            g3 = ((g3 - (.25 * g3_incr)) + ((2 * (.25 * g3_incr)) * np.random.random()))

            eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

            m_min = 1.0
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
            max_mass = int(max_mass*1000)/1000
            m_grid = np.linspace(m_min, max_mass, N)
            m_grid = m_grid[m_grid <= max_mass]

            working_masses = []
            working_radii = []
            for m in m_grid:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            pl.plot(working_masses,working_radii,color="black")
            n_count += 1
        except RuntimeError:
            break

    n_count = 0
    all_eos = lalsim.SimNeutronStarEOSNames
    while n_count < n_n:
        eos = lalsim.SimNeutronStarEOSByName(all_eos[n_count])
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
                break
        pl.plot(working_masses,working_radii,label=all_eos[n_count])
        n_count +=1

    pl.legend()
    pl.xlabel("Mass")
    pl.ylabel("Radius")
    pl.savefig("NICER_mock_data/parameter_space_radii_plots/{}.png".format(label)) # label="APR4_EPP_newbounds_(n_pp)p_(n_n)n"


def plot_permutation_eos_curves(label):
    # Plot all parameter space permutation piecewise parametrized curves of mass-radii, and some named eos mass-radii
    
    pl.clf()
    N = 1000
    p1_incr, g1_incr, g2_incr, g3_incr = .4575, .927, 1.1595, .9285
    log_p1_SI, g1, g2, g3 = 33.4305,3.143,2.6315,2.7315 

    bounds = np.array([p1_incr, g1_incr, g2_incr, g3_incr])
    midpoints = np.array([log_p1_SI, g1, g2, g3])

    upper_bounds = midpoints + bounds
    lower_bounds = midpoints - bounds

    p1s = np.array([lower_bounds[0],midpoints[0],upper_bounds[0]])
    g1s = np.array([lower_bounds[1],midpoints[1],upper_bounds[1]])
    g2s = np.array([lower_bounds[2],midpoints[2],upper_bounds[2]])
    g3s = np.array([lower_bounds[3],midpoints[3],upper_bounds[3]])

    permutations = []
    for p1 in p1s:
        for g1 in g1s:
            for g2 in g2s:
                for g3 in g3s:
                    permutations.append([p1,g1,g2,g3])
    
    permutations.append(list(midpoints)) # So the black line overlaps the gray ones
    for permutation in permutations:
        try:
            log_p1_SI, g1, g2, g3 = permutation
            eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

            m_min = 1.0
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
            max_mass = int(max_mass*1000)/1000
            m_grid = np.linspace(m_min, max_mass, N)
            m_grid = m_grid[m_grid <= max_mass]

            working_masses = []
            working_radii = []
            for m in m_grid:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
                if permutation == list(midpoints): color = "black"
                else: color = "gray"
            pl.plot(working_masses, working_radii, color = color)
        except RuntimeError:
            continue

    eos_list = ["BHF_BBB2", "APR4_EPP", "SK255", "MS1"]
    for eos_name in eos_list:
        try:
            eos = lalsim.SimNeutronStarEOSByName(eos_name)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

            m_min = 1.0
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
            max_mass = int(max_mass*1000)/1000
            m_grid = np.linspace(m_min, max_mass, N)
            m_grid = m_grid[m_grid <= max_mass]

            working_masses = []
            working_radii = []
            for m in m_grid:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            pl.plot(working_masses, working_radii, label=eos_name)
        except RuntimeError:
            continue

    pl.legend()
    pl.xlabel("Mass")
    pl.xlim(left=m_min)
    pl.ylabel("Radius")
    pl.savefig("NICER_mock_data/permutation_plots/{}.png".format(label)) # label = "extreme_bounds"
