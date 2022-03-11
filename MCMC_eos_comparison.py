from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

class param_distro:

    def __init__(self, N, transitions):
        # Hold attributes that used to be global variables

        # List of some of the eos GWXtreme can work with off the cuff
        self.GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2",
                         "SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP",
                         "SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1",
                         "MS1B_PP","MS1_PP","BBB2","AP4","MPA1","MS1B","MS1",
                         "SLY"]

        # List of compatible eos paper found p1,g1,g2,g3 values for
        self.pap_list = ["AP4","BBB2","MPA1","MS1","SLY"]

        # https://arxiv.org/pdf/0812.2163.pdf
        # "Constraints on a phenomenologically parameterized neutron-star 
        # equation of state"
        self.p_eos_val = {"AP4":[33.269,2.830,3.445,3.348]
                         ,"BBB2":[33.331,3.418,2.835,2.832]             
                         ,"MPA1":[33.495,3.446,3.572,2.887]
                         ,"MS1":[33.858,3.224,3.033,1.325]
                         ,"SLY":[33.384,3.005,2.988,2.851]}

        # These are parameters from current (10/30/21) best MCMC runs that best the paper's values
        self.p1_eos_val = {"BBB2":[33.38009664428772,3.350816606537494,2.9128885375511855,2.957866596134232]
                          ,"MS1":[33.92440849651022,3.2385770596974908,3.3780650684604847,2.096509814792764]}

        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

        self.N = N
        self.transitions = transitions

    def eos_to_run(self, eos_list, runs, directory, run0=0, p1_incr=.4575, 
                   g1_incr=.927, g2_incr=1.1595, g3_incr=.9285):

        for eos in eos_list:
            
            for run in range(run0,runs+run0):

                print(eos)
                outputfile = "{}{}_{}.json".format(directory,eos,run)
                self.run_MCMC(eos,outputfile,p1_incr=p1_incr,g1_incr=g1_incr,
                              g2_incr=g2_incr,g3_incr=g3_incr)

    def run_MCMC(self, eos, outputfile, p1_incr=.4575, g1_incr=.927, 
                 g2_incr=1.1595, g3_incr=.9285):
        # For an eos, gets distribution of "best fit" parameters
        
        # If we have the "old" parameters for an eos, use them
        if eos in self.pap_list: log_p1_SI,g1,g2,g3 = self.p1_eos_val[eos]

        else: log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 # defaults

        # Randomly selected "start" parameters
        log_p1_SI = ((log_p1_SI - (.25 * p1_incr)) + ((2 * (.25 * p1_incr)) * np.random.random()))
        g1 = ((g1 - (.25 * g1_incr)) + ((2 * (.25 * g1_incr)) * np.random.random()))
        g2 = ((g2 - (.25 * g2_incr)) + ((2 * (.25 * g2_incr)) * np.random.random()))
        g3 = ((g3 - (.25 * g3_incr)) + ((2 * (.25 * g3_incr)) * np.random.random()))

        eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:

            p1_choice1 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
            g1_choice1 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
            g2_choice1 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
            g3_choice1 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

            try: 

                L1 = self.likelihood(p1_choice1,g1_choice1,g2_choice1,g3_choice1)
                no_errors = True # if L1 doesn't give an error, the while loop will end

            # Can run into ValueError from the use of lal's piecewise function (I think)
            except ValueError: continue
            except RuntimeError: continue

        post_p1 = []
        post_g1 = []
        post_g2 = []
        post_g3 = []
        post_r2 = []
        
        while len(post_p1) <= (self.transitions-1):

            p1_choice2 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
            g1_choice2 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
            g2_choice2 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
            g3_choice2 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

            try: 

                L2 = self.likelihood(p1_choice2,g1_choice2,g2_choice2,g3_choice2) # if L2 gives an error it'll keep trying

            except ValueError: continue
            except RuntimeError: continue

            if L2/L1 >= np.random.random():

                p1_choice1 = p1_choice2
                g1_choice1 = g1_choice2
                g2_choice1 = g2_choice2
                g3_choice1 = g3_choice2

                post_r2.append(L2) # if choice2s are better, append their likelihood
                
            else:

                post_r2.append(L1) # otherwise choice1s are better, so their likelihood is appended instead
            
            # current eos' p1,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            post_p1.append(p1_choice1)
            post_g1.append(g1_choice1)
            post_g2.append(g2_choice1)
            post_g3.append(g3_choice1)
       
        data = {"p1" : post_p1, "g1" : post_g1, "g2" : post_g2, "g3" : post_g3, "r2" : post_r2}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        
    def likelihood(self, log_p1_SI, g1, g2, g3):
        # Produces r2 value between lal and piecewise lambdas

        s, min_mass, max_mass = self.modsel.getEoSInterp_parametrized([log_p1_SI,g1,g2,g3])
        trial_masses = np.linspace(min_mass,max_mass,self.N)
        trial_Lambdas = s(trial_masses)
        trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
        r_val = 1 / np.log(np.sum((self.target_lambdas - trial_lambdas) ** 2)) # r^2 value
               
        return(r_val)

def json_joiner(eos_list, directory, outputfile):
    # Combines component values of like dictionaries from different jsons

    eos_param_distro = {}
    for eos in eos_list: # For each eos

        p1_dist,g1_dist,g2_dist,g3_dist,r2_dist = [],[],[],[],[]
        eos_files = glob.glob(directory+"{}_*".format(eos)) # For each MCMC run of a eos

        for MCMC_run in eos_files:

            with open(MCMC_run, "r") as f:
                data = json.load(f)

            p1_dist += data["p1"]
            g1_dist += data["g1"]
            g2_dist += data["g2"]
            g3_dist += data["g3"]
            r2_dist += data["r2"]

        eos_param_distro.update({eos: {"p1" : p1_dist, "g1" : g1_dist, "g2" : g2_dist, "g3" : g3_dist, "r2" : r2_dist}})

    with open(outputfile, "w") as f:
        json.dump(eos_param_distro, f, indent=2, sort_keys=True)

def global_max_dictionary(eos_list, filename, outputfile):
    # Given parameter distribution dictionary, it produces a global maximum 
    # parameter dictionary per each eos

    with open(filename, "r") as f:
        data = json.load(f)

    m_eos_val = {}
    for eos in eos_list:
            
        max_ind = np.argmax(data[eos]["r2"])

        max_p1 = data[eos]["p1"][max_ind]
        max_g1 = data[eos]["g1"][max_ind]
        max_g2 = data[eos]["g2"][max_ind]
        max_g3 = data[eos]["g3"][max_ind]
        max_r2 = data[eos]["r2"][max_ind]
        m_eos_val.update({eos:[max_p1,max_g1,max_g2,max_g3,max_r2]})

    with open(outputfile,"w") as f:
        json.dump(m_eos_val, f, indent=2, sort_keys=True)
