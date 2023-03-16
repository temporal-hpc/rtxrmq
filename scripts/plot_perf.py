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
files = ["perf-THREADRIPPER-5975WX-ALG1.csv",
         "perf-RTX3090Ti-ALG3.csv",
         "perf-RTX3090Ti-ALG5.csv",
         "perf-RTX3090Ti-constBS-ALG5.csv",
         "perf-RTX3090Ti-constNB-ALG5.csv"]
labels = ["hrmq",
          "alg3",
          "alg5, best config",
          f"alg5, bs=2^{CONSTANT_BS}",
          f"alg5, nb=2^{CONSTANT_NB}"]

def get_data(file):
    hc = pd.read_csv(csv_dir + file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['mean_ns/q'] = hc['ns/q','mean']
    #print(hc)
    return hc

def plot_time(data_frame, lr, dev, saveFlag):
    for i, df in enumerate(data_frame):
        df = df[df['lr'] == lr]
        plt.plot(df['n-exp'], df['mean_ns/q'], label=labels[i])

    plt.legend()
    plt.xlabel("Array size ($2^x$)")
    plt.ylabel("Time per query (ns)")
    plt.title(f"Time vs Array size lr={lr}")

    if saveFlag:
        plt.savefig(f"{plot_dir}time-{dev}-lr{lr}.pdf", dpi=300, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_speedup(data_frame, lr, dev, saveFlag):
    hrmq = data_frame[0]
    hrmq = hrmq[hrmq['lr'] == lr]
    for i, df in enumerate(data_frame[1:], start=1):
        df = df[df['lr'] == lr]
        df_array = np.array(df['mean_ns/q'])
        hrmq_array = np.array(hrmq['mean_ns/q'])[:df_array.size]
        plt.plot(df['n-exp'], hrmq_array/df_array, label=labels[i])

    plt.legend()
    plt.xlabel("Array size ($2^x$)")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Array size lr={lr}")

    if saveFlag:
        plt.savefig(f"{plot_dir}speedup-{dev}-lr{lr}.pdf", dpi=300, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Run with arguments <save_1_0>")
        exit()
    saveFlag = int(sys.argv[1])

    df = []
    for file in files:
        df.append(get_data(file))

    for lr in range(-1,-6,-1):
        plot_time(df, lr, "RTX3090Ti", saveFlag)
        plot_speedup(df, lr, "RTX3090Ti", saveFlag)
