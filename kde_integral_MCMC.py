from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json
import scipy.stats as st

class param_distro:

    def __init__(self, N, transitions, datafile):
        
        self.N = N
        self.transitions = transitions
        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

        self.masses, self.radii = np.loadtxt(datafile, unpack=True) # Mass Radius distribution of mock data
        pairs = np.vstack([self.masses, self.radii])
        self.kernel = st.gaussian_kde(pairs)
    
    def run_MCMC(self, outputfile, p1_incr=.4575, g1_incr=.927, g2_incr=1.1595, g3_incr=.9285):

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
        m_min = min(self.masses)
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

        integral = np.sum(self.kernel(np.vstack([working_masses, working_radii]))*np.diff(working_masses)[0])
        return(integral)
    
    def global_max_dictionary(self, filename, outputfile):
    # Given parameter distribution dictionary, it produces a global maximum
    # parameter dictionary per each eos

        with open(filename, "r") as f:
            data = json.load(f)

        m_eos_val = {}

        max_ind = np.argmax(data["l"])

        max_p1 = data["p1"][max_ind]
        max_g1 = data["g1"][max_ind]
        max_g2 = data["g2"][max_ind]
        max_g3 = data["g3"][max_ind]
        max_l = data["l"][max_ind]
        m_eos_val.update({"eos":[max_p1,max_g1,max_g2,max_g3,max_l]})
        self.m_eos_val = [max_p1,max_g1,max_g2,max_g3,max_l]

        with open(outputfile,"w") as f:
            json.dump(m_eos_val, f, indent=2, sort_keys=True)

    def plot_m_eos_val_on_kde(self, name):
        # Plot the heat and kde of the eos' radii distribution

        m_min, m_max = min(self.masses), max(self.masses) # 1.001069, 2.157369
        r_min, r_max = min(self.radii), max(self.radii) # 9242.634454, 13119.70321

        # Perform the kernel density estimate
        mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arraysgg
        positions = np.vstack([mm.ravel(), rr.ravel()])
        f = np.reshape(self.kernel(positions).T, mm.shape)

        fig = pl.figure()
        ax = fig.gca()
        ax.set_xlim(m_min, m_max)
        ax.set_ylim(r_min, r_max)

        ax.pcolormesh(mm, rr, f)
        ax.set_xlabel('Mass')
        ax.set_ylabel('Radius')

        log_p1_SI, g1, g2, g3, _ = self.m_eos_val
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        m_min = np.min(self.masses)
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        m_grid = np.linspace(m_min, max_mass, self.N)
        m_grid = m_grid[m_grid <= max_mass]

        working_masses = []
        working_radii = []
        for m in m_grid:
            try:
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                working_masses.append(m)
                working_radii.append(rr)
            except RuntimeError:
                break
        pl.plot(working_masses,working_radii,label=self.m_eos_val)

        pl.savefig("NICER_mock_data/MCMC_results/mass_radii_{}.png".format(name))
