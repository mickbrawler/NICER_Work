import numpy as np
import emcee as mc
import h5py
import lal

def random_choice(rm_file, N=50000, label="N50k_J0740"):
    # ADD COMPACTNESS CALCULATION AND MAKE AS SEPARATE FILES
    # Randomly chooses N points from mr posterior

    km_radii, masses, _ = np.loadtxt(rm_file).T
    m_radii = km_radii*1000
    cc = masses*lal.MRSUN_SI/m_radii

    mr_samples = np.array([masses,m_radii]).T # rearrange Miller's m-r distribution to be in m, r (meters)
    mc_samples = np.array([masses,cc]).T
    
    random_positions = [np.random.randint(0,len(mr_samples))]
    while len(random_positions) < N:
        random = np.random.randint(0,len(mr_samples))
        appearances = np.sum(random == np.array(random_positions))
        if appearances == 0:
            random_positions.append(random)

    shortened_mr_samples = mr_samples[random_positions]
    shortened_mc_samples = mc_samples[random_positions]

    np.savetxt("run_data/"+label+"_RM.txt", shortened_mr_samples)
    np.savetxt("run_data/"+label+"_CM.txt", shortened_mc_samples)

def thinner(spectral_file, outputfile, burnin=1000):
    # Shortens large spectral sample file resulting from emcee run

    f1 = h5py.File(spectral_file,'r+')
    samples = f1['chains']
    ls = f1['logp']
    thinning = int(max(mc.autocorr.integrated_time(samples))/2.)

    shortened_data = []
    for i in range(burnin,len(samples),thinning):
        for j in range(len(samples[0])):
            g1,g2,g3,g4 = samples[i,j,:]
            l = ls[i,j]
            shortened_data.append([g1,g2,g3,g4,l])

    np.savetxt(outputfile, shortened_data)

