from GWXtreme import eos_prior as ep
from GWXtreme import eos_model_selection as ems
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

class parametric_EoS:
    # Uses c-m or r-m distribution to get [g1,g2,g3,g4] distribution.

    def __init__(self, mc_mr_file, N=1000, spectral=True):
        '''
        Constructor that sets up prior bounds for selected parametric model
        and sets up interpolent/kernel thats used to calculate each sample's
        likelihood. The m-c/m-r txt file's format needs to be in the following
        format:
        #mass       radius/compactness
        ...         ...
        ...         ...
        max_mass    ...
        '''

        if spectral: self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
        else: self.priorbounds = {'logP':{'params':{"min":32.6,"max":33.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
        self.spectral = spectral
        self.N = N

        masses, compacts_rads = np.loadtxt(mc_mr_file, unpack=True) # compacts_rads is variable that interchangebly holds compactnesses or radii values
        pairs1 = np.vstack([masses, compacts_rads])
        self.kernel_EM = st.gaussian_kde(pairs1)
        self.modsel = ems.Model_selection(posteriorFile="./run_data/posterior_samples/posterior_samples_narrow_spin_prior.dat", spectral=True)

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
                working_rads = []
                working_compacts = []
                for m in m_grid:
                    try:
                        r = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                        c = m*lal.MRSUN_SI/r
                        working_masses.append(m)
                        working_rads.append(r)
                        working_compacts.append(c)
                    except RuntimeError:
                        continue

                K_EM = np.sum(np.array(self.kernel_EM(np.vstack([working_masses, working_compacts])))) # posterior
                K_GW = self.modsel.eos_evidence(parameters)
                dm = np.diff(working_masses)[0]

                return math.log(K_EM*K_GW*dm) # likelihood

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
                raw_samples = sampler.get_chain()
                raw_ls = sampler.get_log_prob()
                flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
            sampler.run_mcmc(p0, sample_size, progress=True)
            raw_samples = sampler.get_chain()
            raw_ls = sampler.get_log_prob()
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

        if self.spectral: model = "spectral"
        else: model = "piecewise"

        outputfile = "run_data/{}_rawsamples_{}.h5".format(model,label)
        f=h5py.File(outputfile,'w')
        f.create_dataset('chains',data=np.array(raw_samples))
        f.create_dataset('logp',data=np.array(raw_ls))
        f.close()

        outputfile = "run_data/{}_samples_{}.txt".format(model,label)
        np.savetxt(outputfile, flat_samples)

