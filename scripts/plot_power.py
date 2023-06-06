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

CONSTANT_BS = 15
CONSTANT_NB = 9
SIZE_MULT = 1.3
plot_dir = "../plots/"
lrLabels = ["","Large $(l,r)$ Range","Medium $(l,r)$ Range", "Small $(l,r)$ Range", "Medium $(l,r)$ Range", "Small $(l,r)$ Range"]
linestyles=[ls['densely dotted'], ls['solid'], ls['densely dashed'], ls['densely dashdotted'], ls['densely dashdotdotted'], ls['dashed']]
#colors=['cornflowerblue','forestgreen','darkslategrey','teal', 'lightseagreen', 'darkorange']
colors=["#EC0B43", "darkslategrey", "#0099ff", "#44AF69", "#ECA400", "#763b82"]
orders=[10, 10, 10, 10, 10, 10]
alphas=[ 1,  1, 0.8, 0.8, 1.0, 0.8]

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Run with arguments <save_1_0> <lr> <ylim1> <ylim2> <outname> <file1> <label1> <reps1> <file2> <label2> <reps2>.. <filen> <labeln> <repsn>")
        print("save_1_0:  1: save file,  0: just show")
        print(f"Example: \npython {sys.argv[0]} 1 0 700 'RTX6000ADA' ../csv-finales/perf-2X-EPYC9654-96C.csv  'HRMQ@192c'\x5c")
        print(f"                         ../csv-finales/perf-RTX6000ADA-ALG5.csv              'RTXRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX6000ADA-ALG7.csv              'LCA'\x5c")
        print(f"                         ../csv-finales/perf-RTX6000ADA-ALG2.csv              'Exhaustive'\x5c")
        exit()

    saveFlag = int(sys.argv[1])
    lr = int(sys.argv[2])
    ylim1 = float(sys.argv[3])
    ylim2 = float(sys.argv[4])
    outName = sys.argv[5]
    files=[]
    labels=[]
    reps=[]
    for i in range(6,len(sys.argv),3):
        files.append(sys.argv[i])
        labels.append(sys.argv[i+1])
        reps.append(int(sys.argv[i+2]))

    #print("Files:", files)
    #print("Labels", labels)
    #print("Reps", reps)
    print(f"Generating power plots for {outName} {lr=}.......",end="")
    sys.stdout.flush()
    title_string = "Power Consumption"
    subtitle_string = "$n=10^8,q=2^{26}$," + f"{lrLabels[-lr]}"

    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(6*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    plt.suptitle(title_string, y=1.05, fontsize=12)
    plt.title(subtitle_string, fontsize=10)
    plt.xlabel("Time [s]",fontsize=12)
    #plt.set_axisbelow(True)
    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')
    # Create a second y-axis on the right
    #ax.yaxis.set_visible(False)
    #ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major',zorder=0)
    plt.ylabel("W",fontsize=12, rotation=0, labelpad=10)
    #plt.xticks([0, 1,10,100,10000, 40000])
    plt.xscale('log')
    plt.xlim(10**-3.5,10**5)
    #plt.xticks([10**-3, 10**-2, 10**-1, 10**0, 10**4])
    for i,file in enumerate(files):
        #print(f"Processing {file=}")
        df = pd.read_csv(file, sep='\s+')
        #print("DATA FILE BEFORE ", i)
        #print(df['acc-time'])
        df["acc-time"] = df["acc-time"]/reps[i]
        minval = df['acc-time'].min()
        df["acc-time"] = df["acc-time"] - minval + 0.001
        #print("DATA FILE AFTER ", i)
        #print(df['acc-time'])
        plt.plot(df['acc-time'], df['power'], label=labels[i], linestyle=linestyles[i],color=colors[i], zorder=orders[i], alpha=alphas[i])
        #print(f"\n\n")
    plt.legend(fontsize=8)
    plt.ylim(ylim1, ylim2)
    if saveFlag:
        plt.savefig(f"{plot_dir}power-{outName}-lr{lr}.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    # generate plots
    print("done")
