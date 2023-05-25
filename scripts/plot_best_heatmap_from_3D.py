import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


def get3Ddata(file):
    hc = pd.read_csv(file)
    #hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['lr-ratio'] = np.round(np.log2(hc['lr'] / hc['n']))
    hc['mean_ns/q'] = hc['ns/q','mean']
    return hc


def heat_map(x, y, df, title, filename, saveFlag, vmax=100):
    df_nb = df.groupby(['n-exp', 'lr-ratio']).agg(np.min).reset_index();
    ax_ticks = {'n-exp' : sorted(df_nb['n-exp'].unique()),
                'lr-ratio' : sorted(df_nb['lr-ratio'].unique())}

    pl = df_nb.pivot(values = 'mean_ns/q', index = y, columns = x)
    #print(pl.shape)

    fig = plt.figure()
    k=0.5
    fig.set_figwidth(6*k)
    fig.set_figheight(4*k)

    minval = df_nb['mean_ns/q'].min()
    maxval = df_nb['mean_ns/q'].max()
    #print(f"{minval=}   {maxval=}")
    plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=minval, vmax=maxval), rasterized=True)
    #plt.colorbar()
    plt.xlabel("Array Size ($n=2^x$)", fontsize=12)
    plt.ylabel("(l,r) range = ($n\cdot 2^y$)", fontsize=12)
    plt.xticks(range(0,26,5), fontsize=10)
    plt.xlim(0,27)
    plt.yticks(range(0,-26,-5), fontsize=10)
    # plt.yticks([i for i in range(5,26,2)])
    plt.title(f"{title}")
    if saveFlag:
        plt.savefig(f"../plots/{filename}-best2D.pdf", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Run with arguments <csv_path> <title> <save_1_0>")
        exit()
    data_path = sys.argv[1]
    fname=Path(data_path).stem
    #print(f"{fname=}")
    title = sys.argv[2]
    saveFlag= int(sys.argv[3])
    #print(f"ARGS:\n\t{data_path=}\n\t{saveFlag}")
    df_3d = get3Ddata(data_path)
    nmin = int(df_3d['n-exp'].min())
    nmax = int(df_3d['n-exp'].max())

    #print("n:", sorted(df_3d['n-exp'].unique()))
    #print("lr:", sorted(df_3d['lr-ratio'].unique()))

    heat_map('n-exp', 'lr-ratio', df_3d, title,fname,saveFlag, vmax=30)
