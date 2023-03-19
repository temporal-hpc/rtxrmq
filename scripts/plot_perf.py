import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

CONSTANT_BS = 15
CONSTANT_NB = 9

plot_dir = "../plots/"
csv_dir = "../csv-finales/"
lrLabels = ["","Large Range (Uniform)","Medium Range (LogNormal)", "Small Range (LogNormal)", "Medium Range (Fixed Fraction)", "Small Range (Fixed Fraction)"]
linestyles=['dotted', 'solid', 'solid','dashed','dotted','dashed']
colors=['cornflowerblue','forestgreen','darkslategrey','teal', 'lightseagreen', 'darkorange']

def get_data(file):
    hc = pd.read_csv(csv_dir + file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['mean_ns/q'] = hc['ns/q','mean']
    #print(hc)
    return hc

def plot_time(data_frame, lr, dev, saveFlag):
    # common plot settings
    k=0.7
    plt.figure(figsize=(6*k,4*k))
    plt.xticks(range(0,26,5), fontsize=10)
    plt.xlim(0,27)

    for i, df in enumerate(data_frame):
        df = df[df['lr'] == lr]
        plt.plot(df['n-exp'], df['mean_ns/q'], label=labels[i], linestyle=linestyles[i],color=colors[i])

    plt.legend(fontsize=6)
    plt.yscale('log')
    plt.xlabel("Array size ($n=2^x$)",fontsize=12)
    plt.ylabel("Time [ms]",fontsize=12)
    plt.title(f"{dev}, {lrLabels[-lr]}")

    if saveFlag:
        plt.savefig(f"{plot_dir}time-{dev}-lr{lr}.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_speedup(data_frame, lr, dev, saveFlag):
    # common plot settings
    k=0.7
    plt.figure(figsize=(6*k,4*k))
    plt.xticks(range(0,26,5), fontsize=10)
    plt.xlim(0,27)


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
    plt.xlabel("Array size ($n=2^x$)",fontsize=12)
    plt.ylabel("Speedup",fontsize=12)
    plt.title(f"{dev}, {lrLabels[-lr]}")

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
    for lr in range(-1,-6,-1):
        if metric=="time":
            plot_time(df, lr, outName, saveFlag)
        if metric=="speedup":
            plot_speedup(df, lr, outName, saveFlag)
    print("done")
