from GWXtreme import eos_prior as ep
from multiprocessing import cpu_count, Pool
from scipy import interpolate
import scipy.stats as st
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import seaborn as sns
import glob
import re
import json
import emcee
import math
import random
import argparse

def eos_radii_posterior(eos_name, N, m_sigma, r_sigma, label):
    # Function that produces the possible masses and radii for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    working_masses = []
    working_radii = []
    N_count = 0
    while N_count < N:
        try:
            m = np.random.normal(1.4, m_sigma, 1)[0] # mean mass of 1.4, standard deviation of m_sigma
            if m < 1.0: continue # mass can't be less than 1 solar mass
            radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam) # actual radius for m given eos
            rr = np.random.normal(radius, r_sigma, 1)[0] # tampered radius to simulate weak signal
            working_masses.append(m)
            working_radii.append(rr)
            N_count += 1
        except RuntimeError: # ??? Error may have resulted from radius calculation (bad m)
            continue

    output = np.vstack((working_masses,working_radii)).T
    ###outputfile = "NICER_mock_data/mass_radii_posterior/mass_radii_{}.txt".format(label) # label="APR4_EPP_N????"
    outputfile = "new_data/mass_radii_{}.txt".format(label)
    np.savetxt(outputfile, output, fmt="%f\t%f")

def plot_radii_scatter(datafile, label):
    # Function to plot scatter of eos' radii

        pl.clf()
        data = np.loadtxt(datafile)
        masses = data[:,0]
        ###radii = data[:,1]
        radii = data[:,1] / 1000

        pl.rcParams.update({"font.size":18})
        pl.figure(figsize=(20,10))
        pl.scatter(masses,radii,s=5)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.title("Radii vs Masses")
        ###pl.savefig("NICER_mock_data/radii_plots/{}.png".format(label)) # label="APR4_EPP_N????"
        pl.savefig("plots/scatter_{}.png".format(label)) # label="APR4_EPP_N????"

def plot_radii_gaussian_kde(datafile, label, EoS=False):
    # Plot the kde of the eos' radii distribution

    pl.clf()
    data = np.loadtxt(datafile)
    m = data[:,0] 
    r = data[:,1] / 1000

#    m_min, m_max = 1, 2.2 # 1.001069, 2.157369
#    r_min, r_max = 9200, 13200 # 9242.634454, 13119.70321

    m_min, m_max = min(m), max(m) # 1.001069, 2.157369
    r_min, r_max = min(r), max(r) # 9242.634454, 13119.70321

    # Perform the kernel density estimate
    mm, rr = np.mgrid[m_min:m_max:1000j, r_min:r_max:1000j] # two 2d arrays
    positions = np.vstack([mm.ravel(), rr.ravel()])
    pairs = np.vstack([m, r])
    kernel = st.gaussian_kde(pairs)
    f = np.reshape(kernel(positions).T, mm.shape)

    fig = pl.figure()
    ax = fig.gca()
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(r_min, r_max)

    ax.pcolormesh(mm, rr, f)
    ax.set_xlabel('Mass')
    ax.set_ylabel('Radius (km)')
    pl.scatter(m,r,s=1,color="black")
    if type(EoS) == str:
        mass, radii = np.loadtxt(EoS).T
        radii = radii / 1000
        pl.plot(mass, radii, color="red")
    pl.title("Mock Radius-Mass Distribution")
    pl.savefig("plots/{}.png".format(label), bbox_inches='tight') # label="APR4_EPP_m(m_sigma)_r(r_sigma)_kde_mesh_scatter"

class parametric_EoS:

    def __init__(self, mr_file, N=1000, spectral=True):
        '''
        Constructor that sets up prior bounds for selected parametric model
        and sets up interpolent/kernel thats used to calculate each sample's
        likelihood. The m-r txt file's format needs to be in the following
        format:
        #mass       radius
        ...         ...
        ...         ...
        max_mass    ...
        '''

        if spectral: self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
        else: self.priorbounds = {'logP':{'params':{"min":32.6,"max":33.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
        self.spectral = spectral

        masses, radii = np.loadtxt(mr_file, unpack=True)
        pairs = np.vstack([masses, radii])
        self.kernel = st.gaussian_kde(pairs)
        self.N = N

    def log_prior(self, parameters):
        '''
        Checks if sample is physical. Returns 0 if physical, and - inf if not.
        parameters  :: Sample's parameters in the form [g1,g2,g3,g4] for example
        '''

        g1_p1, g2_g1, g3_g2, g4_g3 = parameters

        if self.spectral: params = {"gamma1":np.array([g1_p1]),"gamma2":np.array([g2_g1]),"gamma3":np.array([g3_g2]),"gamma4":np.array([g4_g3])}
        else: params = {"logP":np.array([g1_p1]),"gamma1":np.array([g2_g1]),"gamma2":np.array([g3_g2]),"gamma3":np.array([g4_g3])}

        if ep.is_valid_eos(params,self.priorbounds,spectral=self.spectral): return 0
        else: return - np.inf

    def log_likelihood(self, parameters):
        '''
        Finds 1D integral of eos curve over the kde of a mass-radius posterior sample.
        '''

        g1_p1, g2_g1, g3_g2, g4_g3 = parameters

        if self.spectral: params = {"gamma1":np.array([g1_p1]),"gamma2":np.array([g2_g1]),"gamma3":np.array([g3_g2]),"gamma4":np.array([g4_g3])}
        else: params = {"logP":np.array([g1_p1]),"gamma1":np.array([g2_g1]),"gamma2":np.array([g3_g2]),"gamma3":np.array([g4_g3])}

        if ep.is_valid_eos(params,self.priorbounds,spectral=self.spectral):

            try:
                if self.spectral: eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1, g2_g1, g3_g2, g4_g3)
                else: eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(g1_p1, g2_g1, g3_g2, g4_g3)
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

                return math.log(np.sum(np.array(self.kernel(np.vstack([working_masses, working_radii])))*np.diff(working_masses)[0]))
            except RuntimeError: return - np.inf
            except IndexError: return - np.inf

        else: return - np.inf

    def log_posterior(self, parameters):
        '''
        Adds likelihood and prior results.
        '''

        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def n_walker_points(self, walkers):
        '''
        Gets a random starting sample for each walker.
        '''

        bounds = list(self.priorbounds.values())
        points = []
        while len(points) < walkers:

            g1_p1_choice = random.uniform(bounds[0]['params']['min'],bounds[0]['params']['max'])
            g2_g1_choice = random.uniform(bounds[1]['params']['min'],bounds[1]['params']['max'])
            g3_g2_choice = random.uniform(bounds[2]['params']['min'],bounds[2]['params']['max'])
            g4_g3_choice = random.uniform(bounds[3]['params']['min'],bounds[3]['params']['max'])

            parameters = [g1_p1_choice,g2_g1_choice,g3_g2_choice,g4_g3_choice]
            if self.log_likelihood(parameters) != - np.inf: points.append(parameters)

        return np.array(points)

    def run_mcmc(self, label="", sample_size=5000, nwalkers=10, npool=10):
        '''
        Samples in parametric space.
        label           :: End of filenames produced to distinguish runs
        sample_size     :: Amount of samples each walker goes through
        nwalkers        :: Amount of walkers independently working
        npool           :: Amount of cores to use
        '''

        ndim = 4
        p0 = self.n_walker_points(nwalkers)

        if npool > 1:

            with Pool(min(cpu_count(),npool)) as pool:

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior,pool=pool)
                sampler.run_mcmc(p0, sample_size, progress=True)

                flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
            sampler.run_mcmc(p0, sample_size, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

        if self.spectral: model = "spectral"
        else: model = "piecewise"
        outputfile = "./{}_samples_{}.txt".format(model,label)
        np.savetxt(outputfile, flat_samples)

class p_rho_EoS:

    def __init__(self, label="", spectral=True, N=1000):
        '''
        Setting up reused variables as attributes.
        label     ::  End of name for files produced to distinguish runs
        spectral  ::  Which parametric model is used
        N         ::  Length of log pressure grid
        '''

        self.label = label
        self.spectral = spectral
        self.min_log_pressure = 32.0

        if spectral: 
            self.max_log_pressure = 37.06469815599594
            self.model = "spectral"
            self.logp_grid = np.linspace(self.min_log_pressure, self.max_log_pressure, N+1) # Recently changed spectral p array to still be 1000 values, used to be 999 due to snip off
            self.logp_grid = self.logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error
            self.max_log_pressure = max(self.logp_grid[:-1])
        else: 
            self.max_log_pressure = 35.400799437198074
            self.model = "piecewise"
            self.logp_grid = np.linspace(self.min_log_pressure, self.max_log_pressure, N)

    def p_rho_grid(self, samples_file, checker=False):
        '''
        Uses samples file from run on m-r or lambda_tilda-q distribution to 
        compute p vs rho data. Saves the data as a json. The samples file 
        needs to have the following format:
        #g1     g2      g3      g4
        ...     ...     ...     ...
        ...     ...     ...     ...
        checker : If set to True, samples that caused errors will be saved (rare)
        '''

        parametric_samples = np.loadtxt(samples_file).tolist()

        p_densities = {}
        troublesome_psamples = {}
        for lp in self.logp_grid:

            density_grid = []
            troublesome_samples = []
            for sample in parametric_samples:

                g1_p1, g2_g1, g3_g2, g4_g3 = sample
                try:
                    if self.spectral: eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1,g2_g1,g3_g2,g4_g3)
                    else: eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(g1_p1,g2_g1,g3_g2,g4_g3)
                    density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
                except RuntimeError: 
                    troublesome_samples.append(sample)
                    continue # ran into runtime error at some point due to energydensityofpressure function

            troublesome_psamples[lp] = troublesome_samples
            p_densities[lp] = density_grid

        with open("./{}_pressure_densities_{}.json".format(self.model,self.label), "w") as f:
            json.dump(p_densities, f, indent=2, sort_keys=True)

        if checker == True:
            with open("./{}_pressure_troublesome_samples_{}.json".format(self.model,self.label), "w") as f:
                json.dump(troublesome_psamples, f, indent=2, sort_keys=True)

    def confidence_interval(self, p_dens_file, plot=True):
        '''
        Saves logp_grid, lower_bound, median, upper_bound of parametric
        distribution. Can plot them as well.
        '''

        with open(p_dens_file, "r") as f:
            data = json.load(f)

        density_matrix = list(data.values())

        lower_bound = []
        median = []
        upper_bound = []
        counter = 0
        for p_rhos in density_matrix:

            bins, bin_bounds = np.histogram(p_rhos,bins=50,density=True)
            bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
            order = np.argsort(-bins)
            bins_ordered = bins[order]
            bin_cent_ord = bin_centers[order]
            include = np.cumsum(bins_ordered) < 0.9 * np.sum(bins)
            include[np.sum(include)] = True
            lower_bound.append(min(bin_cent_ord[include]))
            median.append(np.median(p_rhos))
            upper_bound.append(max(bin_cent_ord[include]))

        rho_vals = np.array([self.logp_grid, lower_bound, median, upper_bound]).T
        outputfile = "./{}_p_vs_rho_{}.txt".format(self.model,self.label)
        np.savetxt(outputfile, rho_vals)

        if plot:

            pl.clf()

            ax = pl.gca()
            ax.set_xscale("log")

            size = 1
            pl.plot(lower_bound, self.logp_grid, color="blue")
            pl.plot(upper_bound, self.logp_grid, color="blue")
            ax.fill_betweenx(self.logp_grid, lower_bound, x2=upper_bound, color="blue", alpha=0.5)
            pl.plot(median, self.logp_grid, "k--")

            pl.xlim([10**17, 10**19])
            pl.xlabel("Density")
            pl.ylabel("Log Pressure")
            pl.title("Pressure vs Density")
            pl.savefig("./{}_p_vs_rho_{}.png".format(self.model,self.label), bbox_inches='tight')

def plot_parameter_distribution(filename, label, EoS=False):
    # Plots distributions of "detailed" parameter distributions

    data = np.loadtxt(filename)

    sns.set()
    fig, axes = pl.subplots(2,2,figsize=(7,7))

    sns.kdeplot(data[:,0],ax=axes[0,0]).set_title("Gamma 1")
    sns.kdeplot(data[:,1],ax=axes[0,1]).set_title("Gamma 2")
    sns.kdeplot(data[:,2],ax=axes[1,0]).set_title("Gamma 3")
    sns.kdeplot(data[:,3],ax=axes[1,1]).set_title("Gamma 4")

    if EoS:
        APR4_EPP_g1 = .6483014736029169
        APR4_EPP_g2 = .22549530718867078
        APR4_EPP_g3 = -.020071115984931484
        APR4_EPP_g4 = -.0003498568113544248
        axes[0,0].axvline(APR4_EPP_g1)
        axes[0,1].axvline(APR4_EPP_g2)
        axes[1,0].axvline(APR4_EPP_g3)
        axes[1,1].axvline(APR4_EPP_g4)

    pl.tight_layout()
    pl.savefig("plots/dist_kde_{}.png".format(label))

