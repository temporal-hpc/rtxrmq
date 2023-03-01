import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

def get3Ddata(file):
    hc = pd.read_csv(file)
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['lr-ratio'] = np.log2(hc['lr'] / hc['n'])
    return hc

def heat_map(x, y, plane, df, vmax=100):
    col = ""
    for c in ['n-exp', 'nb', 'lr-ratio']:
        if c not in {x, y}:
            col = c
    df_nb = df[df[col] == plane]

    ax_labels = {'n-exp': "Array size (2^x)",
                 'nb' : "Number of blocks (2^x)",
                 'lr-ratio' : "Segment size fraction (2^x)"}
    ax_ticks = {'n-exp' : [x for x in range(16, 27)],
                'nb' : [x for x in range(1,13)],
                'lr-ratio' : [x for x in range(-15, 0)]}

    pl = df_nb.pivot(values = 'ns/q', index = y, columns = x)

    fig = plt.figure()
    fig.set_figwidth(11)
    fig.set_figheight(7)

    plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=.5, vmax=vmax))
    plt.colorbar()
    plt.xlabel(ax_labels[x])
    plt.ylabel(ax_labels[y])
    # plt.yticks([i for i in range(5,26,2)])
    # plt.title("RTX 3090 Ti")
    # plt.savefig(f"../plots/heat_map_3090Ti.pdf", dpi=300, facecolor="#ffffff", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    data_path = sys.argv[1]
    df_3d = get3Ddata(data_path)

    print("n:", df_3d['n-exp'].unique())
    print("nb:", df_3d['nb'].unique())
    print("lr:", df_3d['lr-ratio'].unique())

    heat_map('nb', 'lr-ratio', 24, df_3d, vmax=15)
