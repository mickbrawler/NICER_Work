from GWXtreme import eos_prior as ep
from scipy import interpolate
import scipy.stats as st
import lalsimulation as lalsim
import lal
import numpy as np
import json
import math

# recycled sample_sampler.py code to do quick run for Anarya.
# We need to compute the Z_EM for a given GW-source spectral samples distribution

class Z_EM:

    def __init__(self, mc_file, spectral_file, N=1000):
    # Sample through GW-source samples using mc-evidence as likelihood.
        
        self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}

        masses, compacts_rads = np.loadtxt(mc_file, unpack=True)
        pairs1 = np.vstack([masses,compacts_rads])
        self.kernel1 = st.gaussian_kde(pairs1)
        data = np.loadtxt(spectral_file)
        self.samples = data[:,0:3+1]
        self.GW_Zs = data[:,-1]
        self.N = N

    def compute_each_sample_Z_EM(self, outputfile):

        EM_Zs = []
        for sample in self.samples:

            Z = self.likelihood(sample)
            EM_Zs.append(Z)

        data = {"samples":self.samples.tolist(),"GW_Zs":self.GW_Zs.tolist(),"EM_Zs":EM_Zs}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def likelihood(self, parameters):
    # Uses m-c 1D integral to weight each sample

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
            working_compacts = []
            for m in m_grid:
                try:
                    r = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                    c = m*lal.MRSUN_SI/r
                    working_masses.append(m)
                    working_compacts.append(c)
                except RuntimeError:
                    continue

            K1 = np.array(self.kernel1(np.vstack([working_masses, working_compacts])))

            return math.log(np.sum(K1)*np.diff(working_masses)[0])

        else: return - np.inf

