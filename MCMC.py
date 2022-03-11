from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

class param_distro:

    def __init__(self, N, transitions):
        
        self.N = N
        self.transitions = transitions
        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

        masses, radii = np.loadtxt(datafile, unpack=True) # Mass Radius distribution of mock data
        pairs = np.vstack([masses, radii])
        self.kernel = st.gaussian_kde(pairs)
    
    def run_MCMC(self, eos, outputfile, p1_incr=.4575, g1_incr=.927, 
                 g2_incr=1.1595, g3_incr=.9285):

        log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 # defaults

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:
            
            p1_choice1 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
            g1_choice1 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
            g2_choice1 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
            g3_choice1 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

            try:
                L1 = self.likelihood([p1_choice1,g1_choice1,g2_choice1,g3_choice1])
                no_errors = True # if L1 doesn't give an error, the while loop will end
                
            except RuntimeError: continue

        post_p1 = []
        post_g1 = []
        post_g2 = []
        post_g3 = []
        post_l = []
        while len(post_p1) <= (self.transitions-1):
            
            no_errors = False
            while no_errors == False:
                
                p1_choice2 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
                g1_choice2 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
                g2_choice2 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
                g3_choice2 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

                try: 
                    L2 = self.likelihood([p1_choice2,g1_choice2,g2_choice2,g3_choice2])
                    no_errors = True
                    
                except RuntimeError: continue

            if L2/L1 >= np.random.random():
                p1_choice1 = p1_choice2
                g1_choice1 = g1_choice2
                g2_choice1 = g2_choice2
                g3_choice1 = g3_choice2
                post_l.append(L2) # if choice2s are better, append their likelihood
                
            else:
                post_l.append(L1) # otherwise choice1s are better, so their likelihood is appended instead
            
            # current eos' p1,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            post_p1.append(p1_choice1)
            post_g1.append(g1_choice1)
            post_g2.append(g2_choice1)
            post_g3.append(g3_choice1)
       
        data = {"p1" : post_p1, "g1" : post_g1, "g2" : post_g2, "g3" : post_g3, "l" : post_l}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def likelihood(self, params):
        # Finds integral of eos curve over the kde of a mass-radius posterior sample

        log_p1_SI, g1, g2, g3 = params
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
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

        integral = np.sum(kernel(np.vstack([working_masses, working_radii]))*np.diff(working_masses)[0])
        return(integral)
