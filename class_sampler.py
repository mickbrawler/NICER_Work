import likelihood
import emcee
import numpy as np
import scipy.stats as st

class sampler:

    def __init__(self)

    # parameter space bounds
    self.p1_l_b, self.p1_u_b = 32.973, 33.888
    self.g1_l_b, self.g1_u_b = 2.216, 4.070
    self.g2_l_b, self.g2_u_b = 1.472, 3.791
    self.g3_l_b, self.g3_u_b = 1.803, 3.660

    def obtain_kernel(self, filename):

        masses, radii = np.loadtxt(filename, unpack=True) # Mass Radius distribution of mock data
        pairs = np.vstack([masses, radii])
        self.kernel = st.gaussian_kde(pairs)

    def log_posterior(self, parameters):

        p1, g1, g2, g3 = parameters

        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def log_likelihood(self, parameters):
        
        p1, g1, g2, g3 = parameters
        
        # MCMC code travels outside the parameter space sometimes; if statement is 
        # needed to make sure eos pointer isn't made (resulting in a seg fault)
        if ((self.p1_l_b <= p1 <= self.p1_u_b) & (self.g1_l_b <= g1 <= self.g1_u_b) & 
            (self.g2_l_b <= g2 <= self.g2_u_b) & (self.g3_l_b <= g3 <= self.g3_u_b)):
            try: return np.log(likelihood.likelihood(parameters, self.kernel))
            except RuntimeError: return - np.inf
            except IndexError: return - np.inf
        else:
            return - np.inf

    def log_prior(self, parameters):
        
        p1, g1, g2, g3 = parameters
        
        # If sample values are within bounds, return 0, Else return - infinite
        if ((self.p1_l_b <= p1 <= self.p1_u_b) & (self.g1_l_b <= g1 <= self.g1_u_b) & 
            (self.g2_l_b <= g2 <= self.g2_u_b) & (self.g3_l_b <= g3 <= self.g3_u_b)):
            return 0
        else:
            return - np.inf

    # randomly selected walker starting points
    def n_walker_points(self, walkers):

        p1_incr, g1_incr, g2_incr, g3_incr = .4575, .927, 1.1595, .9285
        log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 # defaults

        points = []
        for walker in range(walkers):

            p1_choice = ((log_p1_SI - p1_incr) +
                        ((2 * p1_incr) * np.random.random()))
            g1_choice = ((g1 - g1_incr) +
                        ((2 * g1_incr) * np.random.random()))
            g2_choice = ((g2 - g2_incr) +
                        ((2 * g2_incr) * np.random.random()))
            g3_choice = ((g3 - g3_incr) +
                        ((2 * g3_incr) * np.random.random()))

            points.append([p1_choice,g1_choice,g2_choice,g3_choice])

        return(np.array(points))

    def run_mcmc(self, label, sample_size=5000):

        ndim, nwalkers = 4, 10
        p0 = self.n_walker_points(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
        sampler.run_mcmc(p0, sample_size)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        outputfile = "emcee_files/runs/{}.txt".format(label)
        np.savetxt(outputfile, flat_samples)

    def run_sampler_per_snr_posterior(self):
        # Reruns code for each snr (m-r) posterior

        # Recreating snr (m-r) distributions' names
        m_sigmas = [.2,.4,.6,.8,1]
        r_sigmas = [250,500,750,1000,1250]
        N = 1000
        Files = []
        File_format = "NICER_mock_data/mass_radii_posterior/mass_radii_APR4_EPP_N{}".format(N)
        for sigma in range(len(m_sigmas)):
            Files.append("{}_m{}_r{}.txt".format(File_format,m_sigmas[sigma],r_sigmas[sigma]))

        count = 0
        for File in Files:
            self.obtain_kernel(File)
            self.run_mcmc("N{}_m{}_r{}".format(N,m_sigmas[count],r_sigmas[count]))
            count += 1
