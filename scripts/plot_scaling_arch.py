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
    title_string = "Energy Efficiency"
    subtitle_string = "$n=10^8,q=2^{26}$"
    plt.suptitle(title_string,x=0.58,y=0.93, fontsize=14)
    plt.title(subtitle_string, fontsize=12)
    plt.xlabel("$(l,r)$ Ranges", fontsize=12)
    plt.ylabel(r'$\frac{RMQs}{Joule}$', rotation=0, fontsize=14, labelpad=13)
    plt.ylim(0.1, 10**12)
    plt.tight_layout()

    # array size and number of queries
    n = 10**8
    q = 2**26

    gpus = 4
    ind = np.arange(gpus)
    width = 0.4

    # TITANRTX, RTX3090Ti, RTX6000ADA
    RTXRMQ = np.array([2.67962,1.191996, 0.521385])
    bar1 = plt.bar(ind, q*HRMQreps/HRMQe, width/2, color = colors[0])

    LCA = np.array([2.714679, 1.253711, 1.226598])
    bar2 = plt.bar(ind+width/2, q*RTXRMQreps/RTXRMQe, width/2, color=colors[1])

    plt.xticks(ind+width*3/4,["TITAN RTX (Turing)", "RTX 3090Ti (Ampere)", "RTX 6000 Ada (Lovelace)", "Future GPU Arch"])
    plt.legend( (bar1, bar2), ('RTXRMQ', 'LCA'), fontsize=8)
    plt.show()
