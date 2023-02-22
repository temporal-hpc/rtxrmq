import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def heat_map(x, y, plane, df, vmax=100):
    col = ""
    for c in ['n', 'nb', 'lr']:
        if c not in {x, y}:
            col = c
    df_nb = df[df[col] == plane]

    ax_labels = {'n': "Array size (2^x)",
                 'nb' : "Number of blocks (2^x)",
                 'lr' : "Segment size fraction (2^x)"}
    ax_ticks = {'n' : [x for x in range(16, 27)],
                'nb' : [x for x in range(1,13)],
                'lr' : [x for x in range(-15, 0)]}

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
    df_3d = pd.read_csv("../data/heat_cube.csv", names=['n','nb','lr','ns/q'])
    
    print("n:", df_3d['n'].unique())
    print("nb:", df_3d['nb'].unique())
    print("lr:", df_3d['lr'].unique())

    heat_map('nb', 'lr', 24, df_3d, vmax=15)
