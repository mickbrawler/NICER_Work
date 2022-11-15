import numpy as np
import matplotlib.pyplot as pl
import json

def plot():
    # Plots confidence intervals on top of each other.

    pl.clf()
       
    merge_logp_grid, merge_lower_bound, merge_median, merge_upper_bound = np.loadtxt("LIGO_presentation/new_data/spectral_p_vs_rho_GW_EM_spectral_sampled_confidence.txt")

    EM_logp_grid, EM_lower_bound, EM_median, EM_upper_bound = np.loadtxt("data/mr_parametric_data/spectral_p_vs_rho_W10_S100000.txt").T

    GW_logp_grid, GW_lower_bound, GW_median, GW_upper_bound = np.loadtxt("data/mr_parametric_data/spectral_GW/spectral_p_vs_rho_GW170817.txt")

    ax1 = pl.gca()
    ax1.set_xscale("log")

    pl.plot(EM_lower_bound, EM_logp_grid, label="EM confidence method", color="blue")
    pl.plot(EM_upper_bound, EM_logp_grid, color="blue")
    ax1.fill_betweenx(EM_logp_grid, EM_lower_bound, x2=EM_upper_bound, color="blue", alpha=0.5)

    pl.plot(GW_lower_bound, GW_logp_grid, label="GW confidence method", color="red")
    pl.plot(GW_upper_bound, GW_logp_grid, color="red")
    ax1.fill_betweenx(GW_logp_grid, GW_lower_bound, x2=GW_upper_bound, color="red", alpha=0.5)

    pl.plot(merge_lower_bound, merge_logp_grid, label="merger confidence method", color="green")
    pl.plot(merge_upper_bound, merge_logp_grid, color="green")
    ax1.fill_betweenx(merge_logp_grid, merge_lower_bound, x2=merge_upper_bound, color="green", alpha=0.5)

    pl.xlim([10**17, 10**19])
    pl.xlabel("Density")
    pl.ylabel("Log Pressure")
    pl.legend()
    pl.title("Pressure vs Density")
    pl.savefig("plots/mr_parametric_plots/spectral_p_vs_rho_EM_GW_comparison.png", bbox_inches='tight')

def plot_intervals(outputfile):
    # Plots confidence intervals on top of each other.

    pl.clf()
    
    Files = ["./data/mr_parametric_data/spectral_p_vs_rho_W10_S100000.txt", "./data/mr_parametric_data/spectral_GW/spectral_p_vs_rho_GW170817.txt", "./LIGO_presentation/new_data/spectral_p_vs_rho_GW_EM_spectral_sampled_confidence.txt"]
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

