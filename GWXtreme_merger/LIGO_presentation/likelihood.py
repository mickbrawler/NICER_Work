import lalsimulation as lalsim
import numpy as np
import lal

def likelihood(params, kernel):
    # Finds integral of eos curve over the kde of a mass-radius posterior sample

    log_p1_SI, g1, g2, g3 = params
    eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_p1_SI, g1, g2, g3)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_min = 1.0
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    max_mass = int(max_mass*1000)/1000
    m_grid = np.linspace(m_min, max_mass, 1000)
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
    
    integral = np.sum(np.array(kernel(np.vstack([working_masses, working_radii])))*np.diff(working_masses)[0])
    return integral
