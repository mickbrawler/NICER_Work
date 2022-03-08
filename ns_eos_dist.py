import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re

# This file's functions will produce the proper distribution for the NICER project

def eos_radii_lambdas(eos_name, N, sigma=.01, name="test"):
    # Function that produces the possible masses, radii, lambdas for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    masses = np.random.normal(1.4, 0.4, N)

    working_masses = []
    working_radii = []
    working_Lambdas = []
    Lambdas = []
    for m in masses:
        try:
            radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            rr = np.random.normal(radius, sigma, 1)[0]
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambdas = (2/3)*kk/(cc**5)
            working_Lambdas.append(Lambdas)
            working_masses.append(m)
            working_radii.append(rr)
        except RuntimeError:
            break

    output = np.vstack((working_masses,working_radii,working_Lambdas)).T
    outputfile = "NICER_mock_data/mean_radii_lambdas/{}.txt".format(name)
    np.savetxt(outputfile, output, fmt="%f\t%f\t%f")

def run_all_eos_radii_lambdas():
    # Function to run multiple eos on eos_radii

    all_eos = lalsim.SimNeutronStarEOSNames
    for eos in all_eos:
        eos_radii_lambdas(eos, 1000, name=eos)

def plot_radii(datafile, name):
    # Function to plot eos' radii

        data = np.loadtxt(datafile)
        masses = data[:,0]
        radii = data[:,1]

        pl.scatter(masses,radii)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.title(name)
        pl.savefig("NICER_mock_data/radii_plots/{}.png".format(name))

def plot_lambdas(datafile, name, N):
    # Function to plot eos' mean produced lambdas and multiple separate eos' lambda curves

    data = np.loadtxt(datafile)
    masses = data[:,0]
    Lambdas = data[:,2]
    pl.scatter(masses,Lambdas,label=name)

    all_eos = lalsim.SimNeutronStarEOSNames
    for eos_name in all_eos:
        eos = lalsim.SimNeutronStarEOSByName(eos_name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        m_min = 1.0
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        masses = np.linspace(m_min, max_mass, N)
        masses = masses[masses <= max_mass]

        working_masses = []
        working_Lambdas = []
        Lambdas = []
        for m in masses:
            try:
                radius = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
                cc = m*lal.MRSUN_SI/rr
                Lambdas = (2/3)*kk/(cc**5)
                working_Lambdas.append(Lambdas)
                working_masses.append(m)
            except RuntimeError:
                break
        pl.plot(masses,working_Lambdas,label=eos_name)
    pl.title("Lambdas vs Masses")
    pl.legend()
    pl.savefig("NICER_mock_data/lambdas_plots/{}.png".format(name))
