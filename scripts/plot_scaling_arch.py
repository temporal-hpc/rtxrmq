import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator

ls = OrderedDict(
    [('solid',               (0, ())),
        ('loosely dotted',      (0, (1, 10))),
        ('dotted',              (0, (1, 5))),
        ('densely dotted',      (0, (1, 1))),

        ('loosely dashed',      (0, (5, 10))),
        ('dashed',              (0, (5, 5))),
        ('densely dashed',      (0, (5, 1))),

        ('loosely dashdotted',  (0, (3, 10, 1, 10))),
        ('dashdotted',          (0, (3, 5, 1, 5))),
        ('densely dashdotted',  (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

SIZE_MULT = 1.3
plot_dir = "../plots/"
lrLabels = ["Large","Medium", "Small"]
linestyles=[ls['densely dotted'], ls['solid'], ls['densely dashed'], ls['densely dashdotted'], ls['densely dashdotdotted'], ls['dashed']]
#colors=['cornflowerblue','forestgreen','darkslategrey','teal', 'lightseagreen', 'darkorange']
colors=["#EC0B43", "darkslategrey", "#0099ff", "#44AF69", "#ECA400", "#763b82"]
orders=[10, 10, 10, 10, 10, 10]
alphas=[ 1,  1, 0.8, 0.8, 1.0, 0.8]

if __name__ == "__main__":
    k=0.7
    fig = plt.figure(figsize=(5*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    title_string = "Scaling Accross GPU Architectures"
    subtitle_string = "$n=10^8,q=2^{26}$"
    plt.suptitle(title_string,x=0.58,y=0.93, fontsize=12)
    plt.title(subtitle_string, fontsize=10)
    plt.xlim(0.9,4.3)
    plt.xlabel("GPU Architectures", fontsize=10)
    plt.ylabel("Speedup over HRMQ", fontsize=10)
    plt.tight_layout()

    # array size and number of queries
    n = 10**8
    q = 2**26

    gpus = 3
    ind = np.arange(gpus)
    width = 0.4

    xvals = np.array([1,2,3,4])
    xticks_labels = ['TITAN RTX\n(Turing, 2018)','RTX 3090Ti\n(Ampere, 2020)','RTX 6000\n(Lovelace, 2022)', 'Future GPU\n(~2024)']
    RTXRMQ_LR1      = np.array([13.479947, 8.013742, 2.726711, 0.85])
    LCA_LR1         = np.array([1.163021, 0.532804, 0.586491, 0.40])
    HRMQ_LR1        = 7.085396

    RTXRMQ_LR2      = np.array([5.090665, 2.735219, 1.191996, 0.45])
    LCA_LR2         = np.array([1.331928, 0.646864, 0.594543, 0.45])
    HRMQ_LR2        = 5.341278

    RTXRMQ_LR3      = np.array([2.679623, 1.203320, 0.521385, 0.2])
    LCA_LR3         = np.array([2.714679, 1.253711, 1.226598, 0.85])
    HRMQ_LR3        = 2.656792

    width=0.7

    plt.axhline(y=1, color=(0.1, 0.1, 0.1, 0.2), linestyle=':')
    #LCA experimental
    plt.plot(xvals[:3], HRMQ_LR1/LCA_LR1[:3], label="LCA", linestyle='-', color=colors[2], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR1/LCA_LR1[:3], marker="$L$", color="black", zorder=10)

    plt.plot(xvals[:3], HRMQ_LR2/LCA_LR2[:3], linestyle='-', color=colors[2], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR2/LCA_LR2[:3], marker=r"$M$", color="black", zorder=10)

    plt.plot(xvals[:3], HRMQ_LR3/LCA_LR3[:3], linestyle='-', color=colors[2], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR3/LCA_LR3[:3], marker=r"$S$", color="black", zorder=10)

    #LCA proyejted
    plt.plot(xvals[2:4], HRMQ_LR1/LCA_LR1[2:4], linestyle='--', color=colors[2], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR1/LCA_LR1[2:4], marker="$L$", color="black", zorder=10)

    plt.plot(xvals[2:4], HRMQ_LR2/LCA_LR2[2:4], linestyle='--', color=colors[2], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR2/LCA_LR2[2:4], marker="$M$", color="black", zorder=10)

    plt.plot(xvals[2:4], HRMQ_LR3/LCA_LR3[2:4], linestyle='--', color=colors[2], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR3/LCA_LR3[2:4], marker="$S$", color="black", zorder=10)


    #RTXRMQ experimental
    plt.plot(xvals[:3], HRMQ_LR1/RTXRMQ_LR1[:3], label="RTXRMQ", color=colors[1], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR1/RTXRMQ_LR1[:3], marker="$L$", color="black", zorder=10)

    plt.plot(xvals[:3], HRMQ_LR2/RTXRMQ_LR2[:3], marker="$M$", color=colors[1], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR2/RTXRMQ_LR2[:3], marker="$M$", color="black", zorder=10)

    plt.plot(xvals[:3], HRMQ_LR3/RTXRMQ_LR3[:3], marker="$S$", color=colors[1], lw=width)
    plt.scatter(xvals[:3], HRMQ_LR3/RTXRMQ_LR3[:3], marker="$S$", color="black", zorder=10)

    #RTXRMQ projected
    plt.plot(xvals[2:4], HRMQ_LR1/RTXRMQ_LR1[2:4], linestyle="--", color=colors[1], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR1/RTXRMQ_LR1[2:4], marker="$L$", color="black", zorder=10)

    plt.plot(xvals[2:4], HRMQ_LR2/RTXRMQ_LR2[2:4], linestyle="--", marker="$M$", color=colors[1], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR2/RTXRMQ_LR2[2:4], marker="$M$", color="black", zorder=10)

    plt.plot(xvals[2:4], HRMQ_LR3/RTXRMQ_LR3[2:4], linestyle="--", marker="$S$", color=colors[1], lw=width)
    plt.scatter(xvals[2:4], HRMQ_LR3/RTXRMQ_LR3[2:4], marker="$S$", color="black", zorder=10)

    plt.legend(fontsize=7, ncol=2)
    plt.yscale('log', base=2)

    plt.xticks(xvals, xticks_labels, fontsize=8)
    #plt.show()
    plot_dir = "../plots/"
    plt.savefig(f"{plot_dir}scaling-arch.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
