import numpy as np
import matplotlib.pyplot as pl
import json
import corner

def plot_intervals(outputfile):
    # Plots differently sourced confidence intervals on top of each other.

    pl.clf()
    
    Files = ["./borrowed_data/spectral_p_vs_rho_W10_S100000.txt", "./new_data/spectral_p_vs_rho_thinned_GW_spectral_confidence.txt", "new_data/spectral_p_vs_rho_GW_EM_spectral_sampled_confidence.txt"]
    labels = ["EM confidence method", "GW confidence method", "merger confidence method"]
    colors = ["blue", "red", "green"]

    for File, label, color in zip(Files,labels,colors):

        logp_grid, lower_bound, median, upper_bound = np.loadtxt(File).T

        ax1 = pl.gca()
        ax1.set_xscale("log")

        pl.plot(lower_bound, logp_grid, label=label, color=color)
        pl.plot(upper_bound, logp_grid, color=color)
        ax1.fill_betweenx(logp_grid, lower_bound, x2=upper_bound, color=color, alpha=0.5)

    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.legend()
    pl.title("Pressure vs Density")
    pl.savefig(outputfile, bbox_inches='tight')

def corner_plots(outputfile):
    # Plots differently sourced corner plots on top of each other.

    pl.clf()
    
    Files = ["./borrowed_data/spectral_samples_W10_S100000.txt","./new_data/thinned_spectral_samples.txt","./new_data/GW_EM_spectral_sampled_samples.json"]
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

