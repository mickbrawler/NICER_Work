import likelihood
import emcee
import numpy as np
import scipy.stats as st

masses, radii = np.loadtxt("NICER_mock_data/mean_radii_lambdas/mass_radii_APR4_EPP_N30000.txt", unpack=True) # Mass Radius distribution of mock data
pairs = np.vstack([masses, radii])
kernel = st.gaussian_kde(pairs)

def log_posterior(parameters):

    p1 , g1, g2, g3 = parameters

    return log_likelihood(parameters) + log_prior(parameters)

def log_likelihood(parameters):
    
    try: return np.log(likelihood.likelihood(parameters, kernel))
    except RuntimeError: return - np.inf

def log_prior(parameters):
    
    p1 , g1, g2, g3 = parameters
    # parameter lower and upper bounds
    p1_l_b, p1_u_b = 32.973, 33.888
    g1_l_b, g1_u_b = 2.216, 4.070
    g2_l_b, g2_u_b = 1.472, 3.791
    g3_l_b, g3_u_b = 1.803, 3.660
    
    # If sample values are within bounds, return 0, Else return - infinite
    if ((p1_l_b <= p1 <= p1_u_b) & (g1_l_b <= g1 <= g1_u_b) & 
        (g2_l_b <= g2 <= g2_u_b) & (g3_l_b <= g3 <= g3_u_b)):
        return 0
    else:
        return - np.inf

