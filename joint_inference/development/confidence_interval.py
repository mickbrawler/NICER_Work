import numpy as np
import matplotlib.pyplot as pl
import json
import corner

def plot_intervals(outputfile):
    # Plots differently sourced confidence intervals on top of each other.

    Files = ["run_data/3rd_r_c_switch/spectral_p_vs_rho_EM_switch_confidence.txt", "run_data/spectral_p_vs_rho_W100_S10000_J0740_XMM_confidence.txt", "run_data/spectral_p_vs_rho_thinned_GW_confidence.txt", "run_data/spectral_p_vs_rho_W100_S10000_GW_J0300_J0740_XMM_hierarchical_confidence.txt"]
    labels = ["J0300", "J0740_XMM", "GW", "EM + GW"]
    colors = ["#a6611a", "#dfc27d", "#80cdc1", "#018571"]

    pl.clf()
    pl.figure(figsize=(9,6))
    pl.rc('font', size=12)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True)
    pl.rc('grid', color='w', linestyle='solid')
    pl.rc('xtick', direction='out', color='gray')
    pl.rc('ytick', direction='out', color='gray')
    pl.rc('patch', edgecolor='#E6E6E6')
    pl.rc('lines', linewidth=2)
    
    for File, label, color in zip(Files,labels,colors):

        logp_grid, lower_bound, median, upper_bound = np.loadtxt(File).T

        ax1 = pl.gca()
        ax1.set_xscale("log")

        pl.plot(lower_bound, logp_grid, label=label, color=color)
        pl.plot(upper_bound, logp_grid, color=color)
        ax1.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color=color, alpha=0.45)

    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.legend()
    pl.title("Pressure vs Density")
    pl.savefig(outputfile, bbox_inches='tight')

def corner_plots(outputfile):
    # Plots differently sourced corner plots on top of each other.

    pl.clf()
    
    Files = ["run_data/presentation/spectral_samples_W10_S100000.txt","run_data/presentation/thinned_spectral_samples.txt","run_data/GW_EM_spectral_sampled_samples.json"]
    labels = ["EM source", "GW source", "merger source"] # not sure if we can implement with corner plot...
    colors = ["blue", "red", "green"]

    counter = 0
    for File, color in zip(Files, colors):

        if File[-3:] == "txt":
            samples = np.loadtxt(File)

        else:
            with open(File,"r") as f:
                samples = np.array(json.load(f)["samples"])

        if counter == 0:
            figure = None

        counter += 1

        figure = corner.corner(samples,color=color,labels=[r"$g1$",r"$g2$",r"$g3$",r"$g4$"],fig=figure)

    pl.savefig(outputfile, bbox_inches='tight')

