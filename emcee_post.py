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

    masses, radii = np.loadtxt("NICER_mock_data/mean_radii_lambdas/mass_radii_APR4_EPP_N30000.txt", unpack=True) # Mass Radius distribution of mock data
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

    # APR4_EPP
    true_parameters = [33.275399244401434, 2.881652000854998, 3.380399843127479, 3.2762125079818984]

    log_p1_SI, g1, g2, g3, _ = global_max(MCMC_distribution_file)
    max_parameters = [log_p1_SI, g1, g2, g3]
    names = ["APR4_EPP", "Max Likelihood"]

    x = 0
    combos = [true_parameters, max_parameters]
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
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            except RuntimeError:
                continue
            except IndexError:
                continue
        pl.plot(working_masses,working_radii,label=names[x])
        x += 1

    pl.legend()
    pl.savefig("emcee_files/plots/mass_radii_{}.png".format(label))

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

    samples = np.loadtxt(filename)

    max_log_pressures = []
    for sample in samples: # Obtain max pressure for each sample

        p1, g1, g2, g3 = sample
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        max_log_pressures.append(max_log_pressure)

    outputfile = "emcee_files/runs/max_p_{}.txt".format(label)
    np.savetxt(outputfile, max_log_pressures)

def p_vs_rho(filename, label):

    samples = np.loadtxt(filename)

    max_log_pressures = []
    for sample in samples: # Obtain max pressure for each sample

        p1, g1, g2, g3 = sample
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3)
        max_log_pressure = np.log10(lalsim.SimNeutronStarEOSMaxPressure(eos))
        max_log_pressures.append(max_log_pressure)

    global_max_log_pressure = max(max_log_pressures) # max maximum pressure
    
    min_log_pressure = 27.5
    logp_grid = np.linspace(min_log_pressure, global_max_log_pressure, 100)
    
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
    higher_bound = []
    trouble_p_vals = []
    counter = 0
    for p_rhos in density_matrix:

        try:
            bins, bin_bounds = np.histogram(p_rhos,bins=50,density=True)
            counter += 1
        except IndexError:
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
        higher_bound.append(max(bin_cent_ord[include]))

    logp_grid = logp_grid[~np.isin(logp_grid,trouble_p_vals)]
    rho_vals = [logp_grid, lower_bound, median, higher_bound]
    outputfile = "emcee_files/runs/p_vs_rho_{}.txt".format(label)
    np.savetxt(outputfile, rho_vals)

def p_vs_rho_plot(filename, label):

    logp_grid, lower_bound, median, higher_bound = np.loadtxt(filename)

    pl.plot(logp_grid, lower_bound, label="lower bound")
    pl.plot(logp_grid, median, label="median")
    pl.plot(logp_grid, higher_bound, label="higher bound")

    pl.xlabel("Log Pressure")
    pl.ylabel("Log Energy-Density")
    pl.title("Pressure vs Energy-Density")
    pl.legend()
    pl.savefig("emcee_files/plots/p_vs_rho_{}.png".format(label))
