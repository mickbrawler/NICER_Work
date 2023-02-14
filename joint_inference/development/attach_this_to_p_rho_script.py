def multiprocess_p_rho_grid(p_index):
    '''
    Same as other p_rho_grid function, but designed for multiprocessing to speed
    up density calculation. Removed error prone sample saving for now though.
    Less things to combine later. Solely for spectral method.

    p_index     : Index of pressures to find densities of.
    '''
    
    samples_file = "../development/run_data/appended_EM_switch_GW_thinned_samples.txt"
#    with open(samples_file,"r") as f:
#        data = json.load(f)
#    parametric_samples = data["samples"]

    parametric_samples = np.loadtxt(samples_file)

    N = 1000
    min_log_pressure = 32.0
    max_log_pressure = 37.06469815599594
    logp_grid = np.linspace(min_log_pressure, max_log_pressure, N+1)
    logp_grid = logp_grid[:-1] # last val is max log pressure. For spectral method, density computation at this pressure causes a runtime error
    logp_grid = logp_grid.reshape([100,10])

    p_densities = {}
    for lp in logp_grid[p_index]:

        density_grid = []
        for sample in parametric_samples:

            g1_p1, g2_g1, g3_g2, g4_g3 = sample
            try:
                eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1,g2_g1,g3_g2,g4_g3)
                density_grid.append(lalsim.SimNeutronStarEOSEnergyDensityOfPressure(10**lp, eos)/lal.C_SI**2)
            except RuntimeError: 
                continue # ran into runtime error at some point due to energydensityofpressure function

        p_densities[lp] = density_grid

    with open("data/multiprocessing/{}.json".format(p_index), "w") as f:
        json.dump(p_densities, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="Picks what pressures to use and label file", type=int)
    args = parser.parse_args()

    multiprocess_p_rho_grid(args.index)
