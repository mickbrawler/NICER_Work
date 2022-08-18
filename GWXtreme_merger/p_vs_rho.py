import lalsimulation as lalsim
import lal
import numpy as np
import matplotlib.pyplot as pl
import math
import random
from scipy import interpolate
import json

class sampler:

    def __init__(self, label="test", spectral=True, N=1000):
        # Setting up reused variables as attributes
        # label     ::  End of name for files produced to distinguish runs
        # spectral  ::  Which parametric model is used
        # N         ::  Length of log pressure grid

        self.label = label
        self.spectral = spectral
        self.min_log_pressure = 32.0

        if spectral: 
            self.max_log_pressure = 37.06469815599594
            self.model = "spectral"
        else: 
            self.max_log_pressure = 35.400799437198074
            self.model = "piecewise"

        self.logp_grid = np.linspace(self.min_log_pressure, self.max_log_pressure, N)
        if self.spectral: 
            self.logp_grid = self.logp_grid[:-1] # last val is max log pressure. For spectral, density calculation at this pressure cause runtime error
            self.max_log_pressure = max(self.logp_grid[:-1])

    def p_rho_grid(self, samples_file):
        # Saves all p vs rho data, and the p vs rho interpolant dictionary as jsons
        # samples_file  ::  File holding samples from run on m-r or lambda_tilda-q distribution

        parametric_samples = np.loadtxt(samples_file)

        p_densities = {}
        self.p_usables = {}
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

            troublesome_psamples.update({lp:troublesome_samples})
            p_densities.update({lp:density_grid})

        with open("p_rho_sampler_files/data/{}_pressure_troublesome_samples_{}.json".format(self.model,self.label), "w") as f:
            json.dump(troublesome_psamples, f, indent=2, sort_keys=True)

        with open("p_rho_sampler_files/data/{}_pressure_densities_{}.json".format(self.model,self.label), "w") as f:
            json.dump(p_densities, f, indent=2, sort_keys=True)

    def get_usables(self, p_dens_file):
        # Uses pressure-density computation result file to produce density min/max/interpolant dictionary.
        # p_dens_files  ::  File holding pressure-density values
        
        with open(p_dens_file,"r") as f:
            p_densities = json.load(f)

        self.p_usables = {}
        for lp in self.logp_grid:

            bins, bin_bounds = np.histogram(p_densities[str(lp)], bins=100, density=True)
            bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
            s = interpolate.interp1d(bin_centers, bins)
            self.p_usables.update({lp:[min(bin_centers),max(bin_centers),s]}) # Get min/max of bin_centers instead of density_grid because of interpolation error
     
    def sampler(self, checkpoint=None, samples=5000):
        # Samples p-rho space to get p-rho distribution
        # checkpoint    ::  If set to path of previous samples file, sampler will continue where that run left off
        # samples       ::  Number of samples used

        # METROPOLIS-HASTINGS
        if checkpoint == None: 
            post_p, post_rho, post_l = [], [], []
            p_choice1 = random.uniform(self.min_log_pressure, self.max_log_pressure)

        else: 
            with open(checkpoint,"r") as f: 
                data = json.load(f)
            
            post_p, post_rho, post_l = data["p"], data["rho"], data["l"]
            p_choice1 = post_p[-1]

        L1, rho_choice1 = self.likelihood(p_choice1)

        while len(post_p) <= (samples - 1):

            p_choice2 = random.uniform(self.min_log_pressure, self.max_log_pressure)
            L2, rho_choice2 = self.likelihood(p_choice2)
            
            if L2/L1 >= np.random.random():
                p_choice1 = p_choice2
                rho_choice1 = rho_choice2
                post_l.append(L2)

            else: post_l.append(L1)

            post_p.append(p_choice1)
            post_rho.append(rho_choice1)

        data = {"p": post_p, "rho": post_rho, "l": post_l}
        with open("p_rho_sampler_files/data/{}_p_rho_samples_{}.json".format(self.model, self.label), "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def likelihood(self, p):
        # Likelihood is weighted by distance of the two grid pressures closest to
        # the p value. It uses these grid pressure's density interpolants to 
        # get a random density's prominence.

        # 2-Nearest-Neighbor
        # The method by which be get the closest neighbors needs more work.
        p1 = self.logp_grid[np.sum(self.logp_grid < p) - 1]
        p2 = self.logp_grid[-np.sum(self.logp_grid > p)]
        min_density_p1, max_density_p1, s_1 = self.p_usables[p1]
        min_density_p2, max_density_p2, s_2 = self.p_usables[p2]
        
        # Use max of min_densities and min of max_densities to avoid being outside interpolation range
        min_density = max([min_density_p1, min_density_p2])
        max_density = min([max_density_p1, max_density_p2]) 
        rho = random.uniform(min_density, max_density)
        d1 = abs(p - p1)
        d2 = abs(p - p2)
        s1 = s_1(rho)
        s2 = s_2(rho)
        L = ((s1 * d2) + (s2 * d1)) / (d1 + d2)

        return L, rho

