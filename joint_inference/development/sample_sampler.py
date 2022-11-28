from GWXtreme import eos_prior as ep
from scipy import interpolate
import scipy.stats as st
import lalsimulation as lalsim
import lal
import numpy as np
import json
import math

class sample_samples:

    def __init__(self, mc_mr_file, spectral_file, N=1000, reweighting=None):
    # Sample through GW-source samples using mr-evidence / mc-evidence as likelihood.
        
        self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}

        masses, compacts_rads = np.loadtxt(mc_mr_file, unpack=True)
        pairs1 = np.vstack([masses,compacts_rads])
        self.kernel1 = st.gaussian_kde(pairs1)
        data = np.loadtxt(spectral_file)
        self.samples = data[:,:-1]
        self.ls = np.exp(data[:,-1])
        self.reweighting = reweighting
        self.N = N

        if type(self.reweighting) == str:
            mr_file = self.reweighting
            masses, rads = np.loadtxt(mr_file, unpack=True)
            pairs2 = np.vstack([masses, rads])
            self.kernel2 = st.gaussian_kde(pairs2)

    def run_MCMC(self, outputfile, burnin=10000):
    # Metropolis algorithm, slightly different
        
        old_sample = self.samples[0]
        E1 = self.likelihood(self.samples[0])
        G1 = self.ls[0]

        post_samples = []
        post_ls = []
        count = 1
        for new_sample in self.samples[1:]:
            
            E2 = self.likelihood(new_sample)
            G2 = self.ls[count]
            
            if (E2/E1)*(G1/G2) >= np.random.random():
                old_sample = new_sample
                E1 = E2
                G1 = G2

            count += 1

            post_samples.append(list(old_sample))
            post_ls.append(E1)

        post_samples = post_samples[burnin:]
        post_ls = post_ls[burnin:]

        data = {"samples":post_samples,"ls":post_ls}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def likelihood(self, parameters):
    # Uses m-c/m-r 1D integral to weight each sample

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

            if type(self.reweighting) == str:
                K1 = np.array(self.kernel1(np.vstack([working_masses, working_rads])))
                K2 = np.array(self.kernel2(np.vstack([working_masses, working_rads])))
            else:
                K1 = np.array(self.kernel1(np.vstack([working_masses, working_compacts])))
                K2 = 1

            return math.log(np.sum((K1/K2)*np.diff(working_masses)[0]))

        else: return - np.inf

