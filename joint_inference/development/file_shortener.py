import numpy as np
import emcee as mc
import h5py

def random_choice(mr_file, N, outputfile):
    # Randomly chooses N points from mr posterior

    km_radii, masses = np.loadtxt(mr_file).T
    samples = np.array([masses,km_radii*1000]).T # rearrange Miller's m-r distribution to be in m, r (meters)
    
    random_positions = [np.random.randint(0,len(samples))]
    while len(random_positions) < N:
        random = np.random.randint(0,len(samples))
        appearances = np.sum(random == np.array(random_positions))
        if appearances == 0:
            random_positions.append(random)

    shortened_samples = samples[random_positions]
    np.savetxt(outputfile, shortened_samples)

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

