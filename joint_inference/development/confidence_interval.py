import numpy as np
import matplotlib.pyplot as pl
import json
import corner

def plot_intervals(outputfiles):
    # Plots differently sourced confidence intervals on top of each other.

#    plotsFiles = [["run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_confidence.txt"],
#                  ["run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0030_confidence.txt"],
#                  ["run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_confidence.txt", "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0030_confidence.txt", "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_J0030_hierarchical_confidence.txt"],
#                  ["run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_confidence.txt", "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0030_confidence.txt", "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0740_XMM_confidence.txt", "run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_GW170817_J0030_J0740_XMM_hierarchical_confidence.txt"]]
    plotsFiles = [["run_data/8th_cutoff_plotting/spectral_p_vs_rho_cutoff_W100_S10000_J0740_XMM_confidence.txt"]]

#    plotsLabels = [["GW170817"], ["J0030"], ["GW170817", "J0030", "Joint"], ["GW170817", "J0030", "J0740", "Joint"]]
    plotsLabels = [["J0740"]]

#    plotsColors = [["#d95f02"], ["#7570b3"], ["#d95f02", "#7570b3", "#000000"], ["#d95f02", "#7570b3", "#1b9e77", "#000000"]]
    plotsColors = [["#1b9e77"]]

    pl.figure(figsize=(12,12))
    pl.rc('font', size=20)
    pl.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    pl.rc('xtick', direction='out', color='black', labelcolor='black')
    pl.rc('ytick', direction='out', color='black', labelcolor='black')
    pl.rc('lines', linewidth=2)
    
    for Files, Labels, Colors, outputfile in zip(plotsFiles,plotsLabels,plotsColors,outputfiles): # increment over each plot file

        pl.clf()
        for File, label, color in zip(Files, Labels, Colors): # increment over each plot

            logp_grid, lower_bound, median, upper_bound = np.loadtxt(File).T

            ax1 = pl.gca()
            pl.plot(np.log10(lower_bound), logp_grid, label=label, color=color)
            pl.plot(np.log10(upper_bound), logp_grid, color=color)
            ax1.fill_betweenx(logp_grid, np.log10(lower_bound), x2=np.log10(upper_bound), color=color, alpha=0.45)

        pl.xlim([16.99, 18.25])
        pl.xlabel(r'$log10(\frac{kg}{m^3})$')
        pl.ylabel(r'$log10(Pa)$')
        pl.legend()
        pl.savefig("plots/AstroClub2023/"+outputfile, bbox_inches='tight')

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

