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
#colors=["#EC0B43", "darkslategrey", "#0099ff", "#44AF69", "#ECA400", "#763b82"]
#orders=[10, 10, 10, 10, 10, 10]
#alphas=[ 1,  1, 0.8, 0.8, 1.0, 0.8]

#         HRMQ (blue)        RTXRMQ            LCA (orange)      Ex (green)   light-orange     purple
colors=["#0099ff",       "darkslategrey",       "darkorange",       "#44AF69",    "#ECA400",    "#763b82"]
orders=[10, 10, 9.0, 9.0, 9.0, 9.0]
alphas=[ 1,  1, 0.8, 0.8, 1.0, 0.4]

if __name__ == "__main__":
    k=0.7
    fig = plt.figure(figsize=(5*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    title_string = "Energy Efficiency"
    subtitle_string = "$n=10^8,q=2^{26}$"
    plt.suptitle(title_string,x=0.58,y=0.93, fontsize=12)
    plt.title(subtitle_string, fontsize=10)
    plt.xlabel("$(l,r)$ Ranges", fontsize=10)
    plt.ylabel(r'$\frac{RMQs}{Joule}$', rotation=0, fontsize=12, labelpad=13)
    plt.ylim(0.1, 10**12)
    plt.tight_layout()

    # array size and number of queries
    n = 10**8
    q = 2**26

    cases = 3
    ind = np.arange(cases)
    width = 0.4

    HRMQreps = np.array([512, 512, 512])
    HRMQe = np.array([141914.36, 97208.23, 49991.334747])
    HRMQw = np.array([611.48, 603.49, 570.89])
    bar1 = plt.bar(ind, q*HRMQreps/HRMQe, width/2, color = colors[0])

    RTXRMQreps = np.array([512, 512, 512])
    RTXRMQe = np.array([28625.93, 13035.02, 5579.14])
    RTXRMQw = np.array([296.15,291.31,288.72])
    bar2 = plt.bar(ind+width/2, q*RTXRMQreps/RTXRMQe, width/2, color=colors[1])

    LCAreps = np.array([512, 512, 512])
    LCAe = np.array([4390.02, 4914.54, 9286.31])
    LCAw = np.array([208.59, 229.15, 215.97])
    bar3 = plt.bar(ind+width/2*2, q*LCAreps/LCAe, width/2, color = colors[2])

    Exreps = np.array([1, 1, 512])
    Exe = np.array([12232800.66, 18371.942032, 36319.25])
    Exw = np.array([279.29, 293.69, 274.10])
    bar4 = plt.bar(ind+width/2*3, q*Exreps/Exe, width/2, color = colors[3])


    plt.yscale('log')
    plt.xticks(ind+width*3/4,[f"{lrLabels[0]}", f"{lrLabels[1]}", f"{lrLabels[2]}"])
    plt.legend( (bar1, bar2, bar3, bar4), ("$\mathrm{REF}_{\mathrm{CPU}}$@192c", 'RTXRMQ', 'LCA', 'Exhaustive'), fontsize=8)
    #plt.show()
    plot_dir = "../plots/"
    plt.savefig(f"{plot_dir}rmq-per-joule.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')