import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import re

# This file's functions produce and display the exact radius for every mass 
# for any equation of state.

def eos_radii(eos_name, N, outputfile="test", m_min=1.0):
    # Function that produces the possible masses and radii for any equation of state

    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]

    working_masses = []
    working_radii = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            working_masses.append(m)
            working_radii.append(rr)
        except RuntimeError:
            break
    
    output = np.vstack((working_masses,working_radii)).T
    outputfile = "NICER_mock_data/{}.txt".format(outputfile)
    np.savetxt(outputfile, output, fmt="%f\t%f")

def run_all_eos_radii():
    # Function to run multiple eos on eos_radii

    all_eos = lalsim.SimNeutronStarEOSNames

    for eos in all_eos:
        eos_radii(eos, 1000, outputfile="{}_radii".format(eos))

def plot_eos_radii(masses, radii, name):
    # Plot the mass and radius of an equation of state
        
        pl.clf()
        pl.plot(masses,radii)
        pl.xlabel("Mass")
        pl.ylabel("Radius")
        pl.title(name)
        pl.savefig("NICER_mock_data/radii_plots/{}.png".format(name))

def plot_all_eos_radii():
    # Function to plot multiple eos' radii

    Files = glob.glob("NICER_mock_data/*.txt")
    for File in Files:
        m_r = np.loadtxt(File)
        masses = m_r[:,0]
        radii = m_r[:,1]
        name = re.split(r"_|/", File)[3]
        plot_eos_radii(masses, radii, name)

# Should create plotting script for all these eos radii values.
# Likely by using os to grab the data from each file in the directory with a .txt ending.
# Then plot it with the appropriate titles and filenames.
