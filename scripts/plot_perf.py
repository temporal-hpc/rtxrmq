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
linestyles=[ls['densely dotted'], ls['densely dashed'], ls['solid'], ls['densely dashdotted'], ls['densely dashdotdotted'], ls['dashed']]
#colors=['cornflowerblue','forestgreen','darkslategrey','teal', 'lightseagreen', 'darkorange']
colors=[ "#EC0B43", "darkslategrey"  , "#0099ff", "#763b82", "#44AF69","#ECA400"]

def get_data(file):
    hc = pd.read_csv(file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['mean_ns/q'] = hc['ns/q','mean']
    #print(hc)
    return hc

def plot_time(data_frame, lr, dev, saveFlag):
    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(6*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    plt.title(f"{dev}, {lrLabels[-lr]}")
    plt.xlabel("Array size (n)",fontsize=12)

    plt.xticks(range(0,26,5), fontsize=12)
    plt.xlim(0,26)

    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')

    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$2^{%.0f}$"))
    # Create a second y-axis on the right
    ax2 = ax.twinx()

    ax.yaxis.set_visible(False)

    ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')

    plt.ylabel("Time [ms]",fontsize=12)

    for i, df in enumerate(data_frame):
        df = df[df['lr'] == lr]
        plt.plot(df['n-exp'], df['mean_ns/q'], label=labels[i], linestyle=linestyles[i],color=colors[i])

    plt.legend(fontsize=6)
    plt.yscale('log')

    if saveFlag:
        plt.savefig(f"{plot_dir}time-{dev}-lr{lr}.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_speedup(data_frame, lr, dev, saveFlag):
    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(6*k*SIZE_MULT,4*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    plt.ylabel("Speedup",fontsize=12)
    plt.title(f"{dev}, {lrLabels[-lr]}")
    plt.xlabel("Array size (n)",fontsize=12)

    plt.xticks(range(0,26,5), fontsize=12)
    plt.xlim(0,26)
    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')

    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$2^{%.0f}$"))
    # Create a second y-axis on the right
    ax2 = ax.twinx()

    ax.yaxis.set_visible(False)
    plt.ylabel("Speedup",fontsize=12)

    ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.25, axis='both', which='major')
    # AQUI VOY REVISAR ERROR
    hrmq = data_frame[0]
    hrmq = hrmq[hrmq['lr'] == lr]
    for i, df in enumerate(data_frame[1:], start=1):
        df = df[df['lr'] == lr]
        df_array = np.array(df['mean_ns/q'])
        hrmq_array = np.array(hrmq['mean_ns/q'])[:df_array.size]
        #print(f"{df_array.shape=} {labels[i]=}  {hrmq_array.shape=}")
        plt.plot(df['n-exp'], hrmq_array/df_array, label=labels[i], linestyle=linestyles[i], color=colors[i])


    plt.legend(fontsize=6)
    #plt.yscale('log')

    if saveFlag:
        plt.savefig(f"{plot_dir}speedup-{dev}-lr{lr}.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Run with arguments <metric> <save_1_0> <outname> <ref_file> <ref_label> <file1> <label1> <file2> <label2> .. <filen>")
        print("metric: time or speedup")
        print("save_1_0:  1: save file,  0: just show")
        print(f"Example: \npython {sys.argv[0]} speedup 0 'RTX4090' ../csv-finales/perf-THREADRIPPER-5975WX-ALG1.csv  'HRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-ALG3.csv              'RTXRMQ'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-ALG5.csv              'RTXRMQ-B (optimal)'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-constBS-ALG5.csv      'RTXRMQ-B (bs=2^15)'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-constNB-ALG5.csv      'RTXRMQ-B (nb=2^9)'\x5c")
        print(f"                         ../csv-finales/perf-RTX4090-ALG7.csv              'LCA-GPU'\n")

        exit()

    metric = sys.argv[1]
    saveFlag = int(sys.argv[2])
    outName = sys.argv[3]
    files=[]
    labels=[]
    for i in range(4,len(sys.argv),2):
        files.append(sys.argv[i])
        labels.append(sys.argv[i+1])

    #print("Files:", files)
    #print("Labels", labels)
    print(f"Generating {metric} plots for {outName}.......",end="")
    sys.stdout.flush()

    df = []
    for file in files:
        #print(f"Processing {file=}")
        df.append(get_data(file))


    # generate plots
    for lr in range(-1,-4,-1):
        if metric=="time":
            plot_time(df, lr, outName, saveFlag)
        if metric=="speedup":
            plot_speedup(df, lr, outName, saveFlag)
    print("done")
