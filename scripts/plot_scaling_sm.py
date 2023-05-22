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
    title_string = "Scaling Accross SMs (Lovelace Arch)"
    subtitle_string = "$n=10^8,q=2^{26}$"
    plt.suptitle(title_string,x=0.58,y=0.93, fontsize=14)
    plt.title(subtitle_string, fontsize=12)
    plt.xlim(0.9,4.3)
    plt.xlabel("Number of SMs in GPU", fontsize=12)
    plt.ylabel("Speedup over HRMQ", fontsize=14)
    plt.tight_layout()

    # array size and number of queries
    n = 10**8
    q = 2**26

    gpus = 3
    ind = np.arange(gpus)
    width = 0.4

    xvals = np.array([1,2,3,4])
    xticks_labels = ['RTX 4070\n(46 SMs)','RTX 4080\n(76 SMs)','RTX 4090\n(128 SMs)', 'RTX6000 Ada\n(142 SMs)']
    RTXRMQ_LR1      = np.array([, 4.589968, ,])
    LCA_LR1         = np.array([, 0.786467, ,])
    HRMQ_LR1        = 7.085396

    RTXRMQ_LR2      = np.array([, 2.087687, ,])
    LCA_LR2         = np.array([, 0.802718, ,])
    HRMQ_LR2        = 5.341278

    RTXRMQ_LR3      = np.array([, 0.918145, ,])
    LCA_LR3         = np.array([, 1.698885, ,])
    HRMQ_LR3        = 2.656792

    plt.axhline(y=1, color=(0.1, 0.1, 0.1, 0.2), linestyle=':')
    #LCA experimental
    plt.plot(xvals[:3], HRMQ_LR1/LCA_LR1[:3], label="LCA@L", linestyle='-', marker="o", color=colors[3])
    plt.plot(xvals[:3], HRMQ_LR2/LCA_LR2[:3], label="LCA@M", linestyle='-', marker="^", color=colors[3])
    plt.plot(xvals[:3], HRMQ_LR3/LCA_LR3[:3], label="LCA@S", linestyle='-', marker="v", color=colors[3])
    #LCA proyejted
    plt.plot(xvals[2:4], HRMQ_LR1/LCA_LR1[2:4], linestyle='--', marker="o", color=colors[3])
    plt.plot(xvals[2:4], HRMQ_LR2/LCA_LR2[2:4], linestyle='--', marker="^", color=colors[3])
    plt.plot(xvals[2:4], HRMQ_LR3/LCA_LR3[2:4], linestyle='--', marker="v", color=colors[3])


    #RTXRMQ experimental
    plt.plot(xvals[:3], HRMQ_LR1/RTXRMQ_LR1[:3], label="RTXRMQ@L", marker="o", color=colors[1])
    plt.plot(xvals[:3], HRMQ_LR2/RTXRMQ_LR2[:3], label="RTXRMQ@M", marker="^", color=colors[1])
    plt.plot(xvals[:3], HRMQ_LR3/RTXRMQ_LR3[:3], label="RTXRMQ@S", marker="v", color=colors[1])
    #RTXRMQ projected
    plt.plot(xvals[2:4], HRMQ_LR1/RTXRMQ_LR1[2:4], linestyle="--", marker="o", color=colors[1])
    plt.plot(xvals[2:4], HRMQ_LR2/RTXRMQ_LR2[2:4], linestyle="--", marker="^", color=colors[1])
    plt.plot(xvals[2:4], HRMQ_LR3/RTXRMQ_LR3[2:4], linestyle="--", marker="v", color=colors[1])

    plt.legend(fontsize=7, ncol=2)
    plt.yscale('log', base=2)

    plt.xticks(xvals, xticks_labels, fontsize=8)
    plt.show()
