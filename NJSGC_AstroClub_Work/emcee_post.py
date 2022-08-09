import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as st
import scipy
import likelihood
import lalsimulation as lalsim
import lal
# Script meant to handle all the post processing of emcee runs

def save(data, label):
    # Function that will save emcee runs in two formats
    # Tested out different formats for saving

    d_outputfile = "emcee_files/runs/detail_{}.txt".format(label)
    np.savetxt(d_outputfile, data)

    c_outputfile = "emcee_files/runs/clear_{}.txt".format(label)
    np.savetxt(c_outputfile, data, fmt="%f\t%f\t%f\t%f")

def generate_likelihoods(filename, label):
    # Function to produce file with liklihoods included in parameter distribution file

    masses, radii = np.loadtxt("NICER_mock_data/mean_radii_lambdas/mass_radii_APR4_EPP_N30000.txt", unpack=True) # Mass Radius distribution of mock data
    pairs = np.vstack([masses, radii])
    kernel = st.gaussian_kde(pairs)

    p1_l_b, p1_u_b = 32.973, 33.888
    g1_l_b, g1_u_b = 2.216, 4.070
    g2_l_b, g2_u_b = 1.472, 3.791
    g3_l_b, g3_u_b = 1.803, 3.660

    samples = np.loadtxt(filename)
    samples_ls = []
    for sample in samples:

        p1, g1, g2, g3 = sample
        if ((p1_l_b <= p1 <= p1_u_b) & (g1_l_b <= g1 <= g1_u_b) &
            (g2_l_b <= g2 <= g2_u_b) & (g3_l_b <= g3 <= g3_u_b)):
            try: l = np.log(likelihood.likelihood(sample, kernel))
            except RuntimeError: l = - np.inf
            except IndexError: l = - np.inf
        else:
            l = - np.inf
        
        samples_ls.append([p1,g1,g2,g3,l])

    outputfile = "emcee_files/runs/likelihood_{}.txt".format(label)
    np.savetxt(outputfile, samples_ls)

def global_max(filename):
# Given parameter distribution, it produces a global maximum

    samples = np.array(np.loadtxt(filename))

    max_ind = np.argmax(samples[:,4])

    max_p1 = samples[:,0][max_ind]
    max_g1 = samples[:,1][max_ind]
    max_g2 = samples[:,2][max_ind]
    max_g3 = samples[:,3][max_ind]
    max_l = samples[:,4][max_ind]

    return [max_p1,max_g1,max_g2,max_g3,max_l]

def plot_m_eos_val_on_kde(MCMC_distribution_file, label, N=1000):
    # Plot the heat and kde of the eos' radii distribution

    masses, radii = np.loadtxt("NICER_mock_data/mass_radii_posterior/used/mass_radii_APR4_EPP_N1000.txt", unpack=True) # Mass Radius distribution of mock data
    radii = np.array(radii) / 1000
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
    ax.set_ylabel('Radius (km)')
    pl.scatter(masses,radii,s=1,color="black")
    #ax.set_title("Visualization of Mock NICER Data")

    # APR4_EPP
    true_parameters = [33.275399244401434, 2.881652000854998, 3.380399843127479, 3.2762125079818984]

    log_p1_SI, g1, g2, g3, _ = global_max(MCMC_distribution_file)
    max_parameters = [log_p1_SI, g1, g2, g3]
    names = ["APR4_EPP", "Max Likelihood"]

    x = 0
    #combos = [true_parameters, max_parameters]
    combos = [true_parameters]
    for combo in combos:

        log_p1_SI, g1, g2, g3 = combo

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
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam) / 1000
                working_masses.append(m)
                working_radii.append(rr)
            except RuntimeError:
                continue
            except IndexError:
                continue
        pl.plot(working_masses,working_radii,label=names[x],color="red")
        x += 1

    pl.legend()
    pl.title("Visualization of Mock NICER Data")
    pl.savefig("emcee_files/plots/mass_radii_{}.png".format(label), bbox_inches='tight')

def plot_parameter_distribution(filename, label):
    # Plots distributions of "detailed" parameter distributions

    data = np.loadtxt(filename)

    APR4_EPP_p1 = 33.275399244401434
    APR4_EPP_g1 = 2.881652000854998
    APR4_EPP_g2 = 3.380399843127479
    APR4_EPP_g3 = 3.2762125079818984

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))

    sns.kdeplot(data[:,0],ax=axes[0,0]).set_title("Pressure")
    axes[0,0].axvline(APR4_EPP_p1)
    sns.kdeplot(data[:,1],ax=axes[0,1]).set_title("Gamma 1")
    axes[0,1].axvline(APR4_EPP_g1)
    sns.kdeplot(data[:,2],ax=axes[1,0]).set_title("Gamma 2")
    axes[1,0].axvline(APR4_EPP_g2)
    sns.kdeplot(data[:,3],ax=axes[1,1]).set_title("Gamma 3")
    axes[1,1].axvline(APR4_EPP_g3)

    pl.tight_layout()
    pl.savefig("emcee_files/plots/dist_kde_{}.png".format(label))

def max_p_vals(filename, label):
    # Study to see if there is a variance in the max p val across all samples

    samples = np.loadtxt(filename)

    max_log_pressures = []
    for sample in samples: # Obtain max pressure for each sample

        p1, g1, g2, g3 = sample
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        max_log_pressures.append(max_log_pressure)

    outputfile = "emcee_files/runs/max_p_{}.txt".format(label)
    np.savetxt(outputfile, max_log_pressures)

def p_vs_rho(filename, label, N):

    samples = np.loadtxt(filename)

    max_log_pressures = []
    for sample in samples: # Obtain max pressure for each sample

        p1, g1, g2, g3 = sample
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        max_log_pressures.append(max_log_pressure)

    global_max_log_pressure = max(max_log_pressures) # max maximum pressure
    
    min_log_pressure = 32.0
    logp_grid = np.linspace(min_log_pressure, global_max_log_pressure, N)
    
    density_matrix = []
    for lp in logp_grid:

        density_grid = []
        for sample in samples:

            p1, g1, g2, g3 = sample
            eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
            density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)

        density_matrix.append(density_grid)

    lower_bound = []
    median = []
    upper_bound = []
    trouble_p_vals = []
    counter = 0
    for p_rhos in density_matrix:

        try:
            bins, bin_bounds = np.histogram(p_rhos,bins=50,density=True)
            counter += 1
        except IndexError: # Meant to catch error in density (low pressure) region. Doesn't apply anymore
            trouble_p_vals.append(logp_grid[counter])
            counter += 1
            continue

        bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
        order = np.argsort(-bins)
        bins_ordered = bins[order]
        bin_cent_ord = bin_centers[order]
        include = np.cumsum(bins_ordered) < 0.9 * np.sum(bins)
        include[np.sum(include)] = True
        lower_bound.append(min(bin_cent_ord[include]))
        median.append(np.median(p_rhos))
        upper_bound.append(max(bin_cent_ord[include]))

    logp_grid = logp_grid[~np.isin(logp_grid,trouble_p_vals)]
    rho_vals = [logp_grid, lower_bound, median, upper_bound]
    outputfile = "emcee_files/runs/p_vs_rho_{}.txt".format(label)
    np.savetxt(outputfile, rho_vals)

def p_vs_rho_plot(filename, label, N):

    logp_grid, lower_bound, median, upper_bound = np.loadtxt(filename)

    min_log_pressure = 32.0
    eos = lalsim.SimNeutronStarEOSByName("APR4_EPP")
    max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N)

    density_grid = []
    for lp in logp_grid:
        
        density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)

    ax = pl.gca()
    ax.set_xscale("log")

    size = 1
    pl.plot(lower_bound, logp_grid, color="blue")
    pl.plot(upper_bound, logp_grid, color="blue")
    ax.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color="blue", alpha=0.5)
    pl.plot(median, logp_grid, "k--")
    #pl.plot(density_grid, logp_grid, "r-", label="APR4_EPP")

    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.title("Pressure vs Density")
    pl.legend()
    pl.savefig("emcee_files/plots/p_vs_rho_{}.png".format(label), bbox_inches='tight')

def p_vs_rho_plot_multiple(filename, label, N):

    logp_grid, lower_bound, median, upper_bound = np.loadtxt(filename)

    ax = pl.gca()
    ax.set_xscale("log")

    size = 1
    pl.plot(lower_bound, logp_grid, color="blue")
    pl.plot(upper_bound, logp_grid, color="blue")
    ax.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color="blue", alpha=0.25)
    pl.plot(median, logp_grid, "k--")

    min_log_pressure = 32.0
    max_log_pressure = max(logp_grid)

    eos_list = ["APR4_EPP", "SLY", "H4", "KDE0V", "MS1B", "MS1"]
    eos_density_grids = []
    for eos_name in eos_list:

        eos = lalsim.SimNeutronStarEOSByName(eos_name)
        logp_grid = np.linspace(min_log_pressure, max_log_pressure, N)

        pressure_grid = []
        density_grid = []
        for lp in logp_grid:
            
            try:
                density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
                pressure_grid.append(lp)
            except RuntimeError:
                continue
        
        pl.plot(density_grid, pressure_grid, label=eos_name, linewidth=2.0)

    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.title("Pressure vs Density")
    pl.legend()
    pl.savefig("emcee_files/plots/p_vs_rho_{}.png".format(label), bbox_inches='tight')

def snr_radius_error(m_sigmas, r_sigmas, N, sigmas):

    # Recreating snr (m-r) distributions' names
    Files = []
    File_format = "emcee_files/runs/"
    for sigma in range(len(m_sigmas)):
        Files.append("{}N{}_m{}_r{}.txt".format(File_format,N,m_sigmas[sigma],r_sigmas[sigma]))

    eos = lalsim.SimNeutronStarEOSByName("APR4_EPP")
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    true_radius = lalsim.SimNeutronStarRadius(1.4*lal.MSUN_SI, fam)

    std_radii = []
    count = 1
    for File in Files:

        print(count)
        samples = np.loadtxt(File)
        radii = []
        for sample in samples: # Obtain max pressure for each sample

            try:
                p1, g1, g2, g3 = sample
                eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
                fam = lalsim.CreateSimNeutronStarFamily(eos)
                radius = lalsim.SimNeutronStarRadius(1.4*lal.MSUN_SI, fam)
                radii.append(radius)
            except RuntimeError:
                continue
        count += 1

        std_radii.append(np.std(radii))

    quad_data = np.loadtxt("NICER_mock_data/mass_radii_posterior/quadrature_study/APR4_EPP_N{}_sigmas{}.txt".format(N,sigmas))
    quad_m = quad_data[0]
    quad_r = quad_data[1]
    snr = ((np.array(quad_m) ** 2) + (np.array(quad_r) ** 2)) ** .5
    error = np.array(std_radii) / true_radius

    outputfile = "emcee_files/error/N{}_sigmas{}.txt".format(N,sigmas)
    data = [snr, error]
    np.savetxt(outputfile, data)

def snr_error_plotter(filename, N, sigmas):
        
    pl.clf()
    data = np.loadtxt(filename)

    snr = data[0]
    error = data[1]

    pl.rcParams.update({'font.size': 20})
    pl.figure(figsize=(15, 10))
    pl.plot(snr, error)
    pl.xlabel("$\\Delta$")
    pl.ylabel("$\\frac{\\sigma[R_{1.4}]}{R_{1.4}^{\\rm True}}$")
    pl.title("Effect of Mass-Radius Measurement on EoS Inference")
    pl.savefig("emcee_files/plots/error_N{}_sigmas{}.png".format(N,sigmas), bbox_inches='tight')
