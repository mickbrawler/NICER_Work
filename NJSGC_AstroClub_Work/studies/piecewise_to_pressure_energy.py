from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

# lalsimulation preset values
#    // Minimum pressure and energy density of core EOS geom
#     double e0 = 9.54629006e-11; // For reference: 1.1555e35 [cgs]
#     double p0 = 4.43784199e-13; // For reference: 5.3716e32 [cgs]

# bilby spectral decomposition preset values
#    p0=3.01e+33, e0=203000000000000.0
log_p1_SI, g1, g2, g3 = [33.275399244401434, 2.881652000854998, 3.380399843127479, 3.2762125079818984] # APR4_EPP

def eos_pressure_energy_density(log_p1_SI, g1, g2, g3, N=1000, 
                                min_pressure=4.43784199*(10**(-13))):
    # Using parametrized eos pointer to create pressure and energy density distribution

    eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI,g1,g2,g3)

    # minimum pressure: 4.43784199e-13 m^-2
    max_pressure = lalsim.SimNeutronStarEOSMaxPressureGeometerized(eos)
    pressures = list(np.linspace(min_pressure,max_pressure,N))

    energy_densities = []
    for pressure in pressures:
        
        energy_density = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(pressure,eos)
        energy_densities.append(energy_density)
            
    output = np.vstack((pressures,energy_densities)).T
    outputfile = "testing_pressure_energy.txt"
    np.savetxt(outputfile, output, fmt="%1.20f\t%f")

def plot_pressures_energy_densities(datafile):
    # Plot distribution of pressures and energy densities

    pl.clf()

    data = np.loadtxt(datafile)
    pressures = data[:,0]
    energy_densities = data[:,1]

    pl.rcParams.update({"font.size":18})
    pl.figure(figsize=(20,10))
    pl.scatter(energy_densities,pressures,s=5)
    pl.xlabel("Energy Density (m^-2)")
    pl.ylabel("Pressure (m^-2)")
    pl.title("Pressure vs Energy-Density")
    pl.savefig("testing_pressure_energy.png")
