import numpy as np
import emcee as mc
import h5py

def random_choice(mr_file, N, outputfile):
    # Randomly chooses N points from mr posterior

    km_radii, masses = np.loadtxt(mr_file).T
    samples = np.array([masses,km_radii*1000]).T

    shortened_samples = samples[np.random.choice(N,N)]
    np.savetxt(outputfile, shortened_samples)

def thinner(spectral_file, outputfile, burnin=1000):
    # Shortens large spectral sample file resulting from emcee run

    f1 = h5py.File(spectral_file,'r+')
    samples = f1['chains']
    thinning = int(max(mc.autocorr.integrated_time(samples))/2.)

    shortened_samples = []
    for i in range(burnin,len(samples),thinning):
        for j in range(len(samples[0])):
            shortened_samples.append(samples[i,j,:])

    np.savetxt(outputfile, shortened_samples)

