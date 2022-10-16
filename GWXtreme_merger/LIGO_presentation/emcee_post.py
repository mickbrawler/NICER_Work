import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as st
import scipy
import likelihood
import lalsimulation as lalsim
import lal
# Script meant to handle all the post processing of emcee runs

def plot_parameter_distribution(filename, label):
    # Plots distributions of "detailed" parameter distributions

    data = np.loadtxt(filename)

    APR4_EPP_g1 = .6483014736029169
    APR4_EPP_g2 = .22549530718867078
    APR4_EPP_g3 = -.020071115984931484
    APR4_EPP_g4 = -.0003498568113544248

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))

    sns.kdeplot(data[:,0],ax=axes[0,0]).set_title("Gamma 1")
    axes[0,0].axvline(APR4_EPP_g1)
    sns.kdeplot(data[:,1],ax=axes[0,1]).set_title("Gamma 2")
    axes[0,1].axvline(APR4_EPP_g2)
    sns.kdeplot(data[:,2],ax=axes[1,0]).set_title("Gamma 3")
    axes[1,0].axvline(APR4_EPP_g3)
    sns.kdeplot(data[:,3],ax=axes[1,1]).set_title("Gamma 4")
    axes[1,1].axvline(APR4_EPP_g4)

    pl.tight_layout()
    pl.savefig("plots/dist_kde_{}.png".format(label))

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
    ###outputfile = "emcee_files/runs/p_vs_rho_{}.txt".format(label)
    outputfile = "new_data/spectral_p_vs_rho_{}.txt".format(label)
    np.savetxt(outputfile, rho_vals)

def p_vs_rho_plot(filename, label, N=1000):

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
    pl.title("Density-Pressure 90% Confidence Intervals")
    pl.legend()
    ###pl.savefig("emcee_files/plots/p_vs_rho_{}.png".format(label), bbox_inches='tight')
    pl.savefig("plots/spectral_p_vs_rho_{}.png".format(label), bbox_inches='tight')

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
    pl.savefig("plots/p_vs_rho_{}.png".format(label), bbox_inches='tight')

