from GWXtreme import eos_prior as ep
from scipy import interpolate
import scipy.stats as st
import lalsimulation as lalsim
import lal
import numpy as np
import json
import math

class sample_samples:

    def __init__(self, mr_file, spectral_file, N=1000):
    # Sample through GW-source samples using mr-evidence as likelihood.
        
        self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}

        masses, radii = np.loadtxt(mr_file, unpack=True)
        pairs = np.vstack([masses,radii])
        self.kernel = st.gaussian_kde(pairs)
        self.samples = np.loadtxt(spectral_file)
        self.N = N

    def run_MCMC(self, outputfile):
    # Metropolis algorithm, slightly different
        
        old_sample = self.samples[0]
        L1 = self.likelihood(self.samples[0])

        post_samples = []
        post_ls = []
        for new_sample in self.samples[1:]:
            
            L2 = self.likelihood(new_sample)
            
            if L2/L1 >= np.random.random():
                old_sample = new_sample
                L1 = L2

            post_samples.append(list(old_sample))
            post_ls.append(L1)

        data = {"samples":post_samples,"ls":post_ls}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def likelihood(self, parameters):
    # Uses m-r 1D integral to weight each sample

        g1, g2, g3, g4 = parameters
        params = {"gamma1":np.array([g1]),"gamma2":np.array([g2]),"gamma3":np.array([g3]),"gamma4":np.array([g4])}

        if ep.is_valid_eos(params,self.priorbounds):

            eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1, g2, g3, g4)
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

        else: return - np.inf

