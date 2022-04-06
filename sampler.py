import likelihood
import emcee
import numpy as np
import scipy.stats as st

masses, radii = np.loadtxt("NICER_mock_data/mean_radii_lambdas/mass_radii_APR4_EPP_N30000.txt", unpack=True) # Mass Radius distribution of mock data
pairs = np.vstack([masses, radii])
kernel = st.gaussian_kde(pairs)

# parameter lower and upper bounds (separated from functions since its used in 2 of them)
p1_l_b, p1_u_b = 32.973, 33.888
g1_l_b, g1_u_b = 2.216, 4.070
g2_l_b, g2_u_b = 1.472, 3.791
g3_l_b, g3_u_b = 1.803, 3.660

def log_posterior(parameters):

    p1, g1, g2, g3 = parameters

    return log_likelihood(parameters) + log_prior(parameters)

def log_likelihood(parameters):
    
    p1, g1, g2, g3 = parameters
    
    # MCMC code travels outside the parameter space sometimes; if statement is 
    # needed to make sure eos pointer isn't made (resulting in a seg fault)
    if ((p1_l_b <= p1 <= p1_u_b) & (g1_l_b <= g1 <= g1_u_b) & 
        (g2_l_b <= g2 <= g2_u_b) & (g3_l_b <= g3 <= g3_u_b)):
        try: return np.log(likelihood.likelihood(parameters, kernel))
        except RuntimeError: return - np.inf
        except IndexError: return - np.inf
    else:
        return - np.inf

def log_prior(parameters):
    
    p1, g1, g2, g3 = parameters
    
    # If sample values are within bounds, return 0, Else return - infinite
    if ((p1_l_b <= p1 <= p1_u_b) & (g1_l_b <= g1 <= g1_u_b) & 
        (g2_l_b <= g2 <= g2_u_b) & (g3_l_b <= g3 <= g3_u_b)):
        return 0
    else:
        return - np.inf

# randomly selected walker starting points
def n_walker_points(walkers):

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

ndim, nwalkers = 4, 10
p0 = n_walker_points(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(p0, 5000)
#t = sampler.get_autocorr_time()
#flat_samples = sampler.get_chain(discard=t, thin=15, flat=True)
