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

class p_rho_EoS:
    # Uses [g1,g2,g3,g4] distribution to get p-rho confidence interval.

    def __init__(self, label="", spectral=True, N=1000):
        '''
        Setting up reused variables as attributes.
        label     ::  End of name for files produced to distinguish runs
        spectral  ::  Which parametric model is used
        N         ::  Length of log pressure grid
        '''

        self.label = label
        self.spectral = spectral
        self.min_log_pressure = 32.0

        if spectral: 
            self.max_log_pressure = 37.06469815599594
            self.model = "spectral"
            self.logp_grid = np.linspace(self.min_log_pressure, self.max_log_pressure, N+1) # Recently changed spectral p array to still be 1000 values, used to be 999 due to snip off
            self.logp_grid = self.logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error
            self.max_log_pressure = max(self.logp_grid[:-1])
        else: 
            self.max_log_pressure = 35.400799437198074
            self.model = "piecewise"
            self.logp_grid = np.linspace(self.min_log_pressure, self.max_log_pressure, N)

    def p_rho_grid(self, samples_file, checker=False):
        '''
        Uses samples file from run on m-r or lambda_tilda-q distribution to 
        compute p vs rho data. Saves the data as a json. The samples file 
        needs to have the following format:
        #g1     g2      g3      g4
        ...     ...     ...     ...
        ...     ...     ...     ...
        checker : If set to True, samples that caused errors will be saved (rare)
        '''

        parametric_samples = np.loadtxt(samples_file).tolist()

        p_densities = {}
        troublesome_psamples = {}
        for lp in self.logp_grid:

            density_grid = []
            troublesome_samples = []
            for sample in parametric_samples:

                g1_p1, g2_g1, g3_g2, g4_g3 = sample
                try:
                    if self.spectral: eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1,g2_g1,g3_g2,g4_g3)
                    else: eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(g1_p1,g2_g1,g3_g2,g4_g3)
                    density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
                except RuntimeError: 
                    troublesome_samples.append(sample)
                    continue # ran into runtime error at some point due to energydensityofpressure function

            troublesome_psamples[lp] = troublesome_samples
            p_densities[lp] = density_grid

        with open("run_data/{}_pressure_densities_{}.json".format(self.model,self.label), "w") as f:
            json.dump(p_densities, f, indent=2, sort_keys=True)

        if checker == True:
            with open("run_data/{}_pressure_troublesome_samples_{}.json".format(self.model,self.label), "w") as f:
                json.dump(troublesome_psamples, f, indent=2, sort_keys=True)

    def confidence_interval(self, p_dens_file, plot=True, EoS=False, Bins=50):
        '''
        Saves logp_grid, lower_bound, median, upper_bound of parametric
        distribution. Can plot them as well.
        '''

        with open(p_dens_file, "r") as f:
            data = json.load(f)

        density_matrix = list(data.values())

        lower_bound = []
        median = []
        upper_bound = []
        counter = 0
        for p_rhos in density_matrix:

            bins, bin_bounds = np.histogram(p_rhos,bins=Bins,density=True)
            bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
            order = np.argsort(-bins)
            bins_ordered = bins[order]
            bin_cent_ord = bin_centers[order]
            include = np.cumsum(bins_ordered) < 0.9 * np.sum(bins)
            include[np.sum(include)] = True
            lower_bound.append(min(bin_cent_ord[include]))
            median.append(np.median(p_rhos))
            upper_bound.append(max(bin_cent_ord[include]))

        rho_vals = np.array([self.logp_grid, lower_bound, median, upper_bound]).T
        outputfile = "run_data/{}_p_vs_rho_{}.txt".format(self.model,self.label)
        np.savetxt(outputfile, rho_vals)

        if plot:

            pl.clf()

            ax = pl.gca()
            ax.set_xscale("log")

            size = 1
            pl.plot(lower_bound, self.logp_grid, color="blue")
            pl.plot(upper_bound, self.logp_grid, color="blue")
            ax.fill_betweenx(self.logp_grid, lower_bound, x2=upper_bound, color="blue", alpha=0.5)
            pl.plot(median, self.logp_grid, "k--")

            if type(EoS) == str:

                pressure_grid = []
                density_grid = []
                for lp in self.logp_grid:

                    eos = lalsim.SimNeutronStarEOSByName(EoS)
                    try: 
                        density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
                        pressure_grid.append(lp)
                    except RuntimeError: continue

                pl.plot(density_grid,pressure_grid,color="red",label=EoS)

            pl.xlim([10**17, 10**19])
            pl.xlabel("Density")
            pl.ylabel("Log Pressure")
            pl.title("Pressure vs Density")
            pl.legend()
            pl.savefig("plots/{}_p_vs_rho_{}.png".format(self.model,self.label), bbox_inches='tight')

def multiprocess_p_rho_grid(p_index):
    '''
    Same as other p_rho_grid function, but designed for multiprocessing to speed
    up density calculation. Removed error prone sample saving for now though.
    Less things to combine later. Solely for spectral method.
    p_index     : Index of pressures to find densities of.
    '''
    
    samples_file = "../development/run_data/appended_EM_switch_GW_thinned_samples.txt"
#    with open(samples_file,"r") as f:
#        data = json.load(f)
#    parametric_samples = data["samples"]

    parametric_samples = np.loadtxt(samples_file)

    N = 1000
    min_log_pressure = 32.0
    max_log_pressure = 37.06469815599594
    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N+1)
    logp_grid = logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error
    logp_grid = logp_grid.reshape([100,10])

    p_densities = {}
    for lp in logp_grid[p_index]:

        density_grid = []
        for sample in parametric_samples:

            g1_p1, g2_g1, g3_g2, g4_g3 = sample
            try:
                eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1,g2_g1,g3_g2,g4_g3)
                density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
            except RuntimeError: 
                continue # ran into runtime error at some point due to energydensityofpressure function

        p_densities[lp] = density_grid

    with open("data/multiprocessing/{}.json".format(p_index), "w") as f:
        json.dump(p_densities, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="Picks what pressures to use and label file", type=int)
    args = parser.parse_args()

    multiprocess_p_rho_grid(args.index)
    
def radius_mass_prior(mc_file, outputfile):
    # Constructs gaussian mass & compactness distribution, and then computes radius.
    # This function was made in an attempt to produce gaussian m-r data (serving as 
    # a nonuniform prior) from gaussian m-c data. This m-r data would be used when 
    # reweighting a likelihood from m-r data.
    
    masses, compacts = np.loadtxt(mc_file).T
    median_mass = np.median(masses)
    std_mass = np.std(masses)
    median_compact = np.median(compacts)
    std_compact = np.std(compacts)
    
    N_count = 0
    working_masses = []
    while N_count < len(masses):

        m = np.random.normal(median_mass, std_mass, 1)[0]
        if m < 1.0: continue
        working_masses.append(m)
        N_count += 1

    masses = np.array(working_masses)
    compacts = np.random.normal(median_compact, std_compact, len(masses))
    rads = masses*lal.MRSUN_SI/compacts

    data = np.array([masses, rads]).T
    np.savetxt(outputfile, data)
    
