import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter

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
orders=[10, 10, 0.0, 0.0, 5.0, 0.0]
alphas=[ 1,  1, 0.8, 0.8, 1.0, 0.8]

def get_data(file):
    hc = pd.read_csv(file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['q-exp'] = np.log2(hc['q'])
    #hc['n-exp'] = hc['n']
    #print(hc['n-exp'])
    #hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['mean_ns/q'] = hc['ns/q','mean']
    #print(hc)
    return hc

def plot_time(data_frame, lr, dev, saveFlag):
    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(6*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    plt.title(f"{dev}, "+r"$n=2^{24}$"+f", {lrLabels[-lr]}")
    plt.xlabel("Number of Queries",fontsize=12)
    plt.xticks(range(0,26,5), fontsize=12)
    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')
    # Create a second y-axis on the right
    ax2 = ax.twinx()
    ax.yaxis.set_visible(False)
    ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')
    plt.ylabel("$\\frac{ns}{q}$",fontsize=12, rotation=0)
    for i, df in enumerate(data_frame):
        df = df[df['lr'] == lr]
        plt.plot(df['q-exp'], df['mean_ns/q'], label=labels[i], linestyle=linestyles[i],color=colors[i], zorder=orders[i], alpha=alphas[i])

    plt.legend(fontsize=6)
    plt.yscale('log')
    #plt.xscale('log')
    if ylim1 >= 0 and ylim2 >= 0:
        plt.ylim(ylim1, ylim2)

    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$2^{%.0f}$"))
    if saveFlag:
        plt.savefig(f"{plot_dir}qperfexp-{dev}-lr{lr}.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Run with arguments <lr> <save_1_0> <ylim1> <ylim2> <outname> <ref_file> <ref_label> <file1> <label1> <file2> <label2> .. <filen>")
        print("lr: -1 (large), -2 (medium), -3 (small)")
        print("save_1_0:  1: save file,  0: just show")
        print(f"Example: \npython {sys.argv[0]} speedup 0 'RTX4090' ../csv-finales/perf-TRPRO-5975WX-QPERF-ALG1.csv  'HRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-QPERF-ALG5.csv              'RTXRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-QPERF-constBS--ALG5.csv     'RTXRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-QPERF-constNB--ALG5.csv     'RTXRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-QPERF-ALG7.csv              'LCA-GPU'\n")

        exit()

    lr = int(sys.argv[1])
    saveFlag = int(sys.argv[2])
    ylim1 = float(sys.argv[3])
    ylim2 = float(sys.argv[4])
    outName = sys.argv[5]
    files=[]
    labels=[]
    for i in range(6,len(sys.argv),2):
        files.append(sys.argv[i])
        labels.append(sys.argv[i+1])

    #print("Files:", files)
    #print("Labels", labels)
    print(f"Generating {lr=} plots for {outName}.......",end="")
    sys.stdout.flush()

    df = []
    for file in files:
        #print(f"Processing {file=}")
        df.append(get_data(file))

    # generate plots
    plot_time(df, lr, outName, saveFlag)
    print("done")
