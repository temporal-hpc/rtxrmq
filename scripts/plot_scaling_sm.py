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
    # array size and number of queries
    n = 60*10**6
    q = 2**26
    fig = plt.figure(figsize=(5*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    title_string = "Scaling Accross Lovelace Architecture"
    subtitle_string = r"$n=60$M, $q=2^{26}$"
    plt.suptitle(title_string,x=0.58,y=0.93, fontsize=14)
    plt.title(subtitle_string, fontsize=12)
    plt.xlim(59,143)
    plt.xlabel("# SMs in GPU", fontsize=12)
    plt.ylabel("Relative Speedup", fontsize=14)
    plt.tight_layout()


    xvals = np.array([60,76,128,142])
    xticks_labels = ['60\nRTX 4070 Ti','76\nRTX 4080','128\nRTX 4090', '142\nRTX 6000 Ada']
    RTXRMQ_LR1      = np.array([5.304304, 4.036054, 2.347052, 2.233117])
    RTXRMQ_LR1_REF  = np.array([5.304304, 5.304304, 5.304304, 5.304304])

    LCA_LR1         = np.array([1.015928, 0.770739, 0.500865, 0.559797])
    LCA_LR1_REF     = np.array([1.015928, 1.015928, 1.015928, 1.015928])
    HRMQ_LR1        = 6.063491

    RTXRMQ_LR2      = np.array([2.245298, 1.692734, 1.228715, 1.122089])
    RTXRMQ_LR2_REF  = np.array([2.245298, 2.245298, 2.245298, 2.245298])

    LCA_LR2         = np.array([1.038174, 0.788144, 0.512562, 0.574361])
    LCA_LR2_REF     = np.array([1.038174, 1.038174, 1.038174, 1.038174])
    HRMQ_LR2        = 4.038768

    RTXRMQ_LR3      = np.array([1.132042, 0.765076, 0.489901, 0.449639])
    RTXRMQ_LR3_REF  = np.array([1.132042, 1.132042, 1.132042, 1.132042])

    LCA_LR3         = np.array([2.204702, 1.629609, 1.045209, 1.130855])
    LCA_LR3_REF     = np.array([2.204702, 2.204702, 2.204702, 2.204702])
    HRMQ_LR3        = 2.112656

    plt.axhline(y=1, color=(0.1, 0.1, 0.1, 0.2), linestyle=':')
    #LCA experimental
    plt.plot(xvals, LCA_LR1_REF/LCA_LR1, label="LCA@L", linestyle='-', marker="o", color=colors[3])
    plt.plot(xvals, LCA_LR2_REF/LCA_LR2, label="LCA@M", linestyle='-', marker="^", color=colors[3])
    plt.plot(xvals, LCA_LR3_REF/LCA_LR3, label="LCA@S", linestyle='-', marker="v", color=colors[3])

    #RTXRMQ experimental
    plt.plot(xvals, RTXRMQ_LR1_REF/RTXRMQ_LR1, label="RTXRMQ@L", marker="o", color=colors[1])
    plt.plot(xvals, RTXRMQ_LR2_REF/RTXRMQ_LR2, label="RTXRMQ@M", marker="^", color=colors[1])
    plt.plot(xvals, RTXRMQ_LR3_REF/RTXRMQ_LR3, label="RTXRMQ@S", marker="v", color=colors[1])

    plt.legend(fontsize=7, ncol=2)
    #plt.yscale('log', base=2)

    plt.xticks(xvals, xticks_labels, fontsize=7)
    #plt.show()
    plot_dir = "../plots/"
    plt.savefig(f"{plot_dir}scaling-sm.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
