from GWXtreme import eos_prior as ep
from multiprocessing import cpu_count, Pool
from scipy import interpolate
import lalsimulation as lalsim
import lal
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st
import json
import emcee
import math
import random
import argparse

class parametric_EoS:

    def __init__(self, mr_file, N=1000, spectral=True):
        '''
        Constructor that sets up prior bounds for selected parametric model
        and sets up interpolent/kernel thats used to calculate each sample's
        likelihood. The m-r txt file's format needs to be in the following
        format:

        #mass       radius
        ...         ...
        ...         ...
        max_mass    ...
        '''

        if spectral: self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
        else: self.priorbounds = {'logP':{'params':{"min":32.6,"max":33.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
        self.spectral = spectral

        masses, radii = np.loadtxt(mr_file, unpack=True)
        pairs = np.vstack([masses, radii])
        self.kernel = st.gaussian_kde(pairs)
        self.N = N

    def log_prior(self, parameters):
        '''
        Checks if sample is physical. Returns 0 if physical, and - inf if not.

        parameters  :: Sample's parameters in the form [g1,g2,g3,g4] for example
        '''

        g1_p1, g2_g1, g3_g2, g4_g3 = parameters

        if self.spectral: params = {"gamma1":np.array([g1_p1]),"gamma2":np.array([g2_g1]),"gamma3":np.array([g3_g2]),"gamma4":np.array([g4_g3])}
        else: params = {"logP":np.array([g1_p1]),"gamma1":np.array([g2_g1]),"gamma2":np.array([g3_g2]),"gamma3":np.array([g4_g3])}

        if ep.is_valid_eos(params,self.priorbounds,spectral=self.spectral): return 0
        else: return - np.inf

    def log_likelihood(self, parameters):
        '''
        Finds 1D integral of eos curve over the kde of a mass-radius posterior sample.
        '''

        g1_p1, g2_g1, g3_g2, g4_g3 = parameters

        if self.spectral: params = {"gamma1":np.array([g1_p1]),"gamma2":np.array([g2_g1]),"gamma3":np.array([g3_g2]),"gamma4":np.array([g4_g3])}
        else: params = {"logP":np.array([g1_p1]),"gamma1":np.array([g2_g1]),"gamma2":np.array([g3_g2]),"gamma3":np.array([g4_g3])}

        if ep.is_valid_eos(params,self.priorbounds,spectral=self.spectral):

            try:
                if self.spectral: eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1, g2_g1, g3_g2, g4_g3)
                else: eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(g1_p1, g2_g1, g3_g2, g4_g3)
                fam = lalsim.CreateSimNeutronStarFamily(eos)
                m_min = 1.0
                max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
                max_mass = int(max_mass*1000)/1000
                m_grid = np.linspace(m_min, max_mass, self.N)
                m_grid = m_grid[m_grid <= max_mass]

                working_masses = []
                working_radii = []
                for m in m_grid:
                    try:
                        r = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                        working_masses.append(m)
                        working_radii.append(r)
                    except RuntimeError:
                        continue

                return math.log(np.sum(np.array(self.kernel(np.vstack([working_masses, working_radii])))*np.diff(working_masses)[0]))
            except RuntimeError: return - np.inf
            except IndexError: return - np.inf

        else: return - np.inf

    def log_posterior(self, parameters):
        '''
        Adds likelihood and prior results.
        '''

        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def n_walker_points(self, walkers):
        '''
        Gets a random starting sample for each walker.
        '''

        bounds = list(self.priorbounds.values())
        points = []
        while len(points) < walkers:

            g1_p1_choice = random.uniform(bounds[0]['params']['min'],bounds[0]['params']['max'])
            g2_g1_choice = random.uniform(bounds[1]['params']['min'],bounds[1]['params']['max'])
            g3_g2_choice = random.uniform(bounds[2]['params']['min'],bounds[2]['params']['max'])
            g4_g3_choice = random.uniform(bounds[3]['params']['min'],bounds[3]['params']['max'])

            parameters = [g1_p1_choice,g2_g1_choice,g3_g2_choice,g4_g3_choice]
            if self.log_likelihood(parameters) != - np.inf: points.append(parameters)

        return np.array(points)

    def run_mcmc(self, label="", sample_size=5000, nwalkers=10, npool=10):
        '''
        Samples in parametric space.

        label           :: End of filenames produced to distinguish runs
        sample_size     :: Amount of samples each walker goes through
        nwalkers        :: Amount of walkers independently working
        npool           :: Amount of cores to use
        '''

        ndim = 4
        p0 = self.n_walker_points(nwalkers)

        if npool > 1:

            with Pool(min(cpu_count(),npool)) as pool:

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior,pool=pool)
                sampler.run_mcmc(p0, sample_size, progress=True)

                flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
            sampler.run_mcmc(p0, sample_size, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

        if self.spectral: model = "spectral"
        else: model = "piecewise"
        outputfile = "./{}_samples_{}.txt".format(model,label)
        np.savetxt(outputfile, flat_samples)


class p_rho_EoS:

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

        with open("new_data/{}_pressure_densities_{}.json".format(self.model,self.label), "w") as f:
            json.dump(p_densities, f, indent=2, sort_keys=True)

        if checker == True:
            with open("new_data/{}_pressure_troublesome_samples_{}.json".format(self.model,self.label), "w") as f:
                json.dump(troublesome_psamples, f, indent=2, sort_keys=True)

