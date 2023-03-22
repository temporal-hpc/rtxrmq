import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


def get3Ddata(file):
    hc = pd.read_csv(file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['lr-ratio'] = np.round(np.log2(hc['lr'] / hc['n']))
    hc['mean_ns/q'] = hc['ns/q','mean']
    return hc

def get_label(col, ax):
    unit = f"2^{ax}"
    #label = ""
    #match col:
    #    case 'n-exp':
    #        label += "Array size"
    #    case 'nb' :
    #        label += "Number of blocks"
    #    case 'lr-ratio' :
    #        label += "Segment size fraction"
    #return label + " " + unit
    label = ""
    match col:
        case 'n-exp':
            label += f"Array size ($n={unit}$)"
        case 'nb' :
            label += f"Number of blocks ($n_b={unit}$)"
        case 'lr-ratio' :
            label += f"Query Range ($|q| = n{unit}$)"
    return label

def get_title(title, x, y, col, plane):
    #ax_name = {'n-exp' : "n",
    #            'nb' : "#Blocks",
    #            'lr-ratio' : "Query length"}
    #if plane is None:
    #    return f"{title}, {ax_name[y]} vs {ax_name[x]}"
    #return f"{title}, {ax_name[y]} vs {ax_name[x]}, {ax_name[col]}=2^{plane}"
    ax_name = {'n-exp' : "n",
                'nb' : "#Blocks",
                'lr-ratio' : "Query Range"}
    if plane is None:
        return f"{title}"
    return f"{title}, {ax_name[col]}=2^{plane}"



def heat_map(x, y, plane, df, title, filename, saveFlag, vmax=100):
    col = ""
    for c in ['n-exp', 'nb', 'lr-ratio']:
        if c not in {x, y}:
            col = c

    df_nb = df if plane is None else df[df[col] == plane]

    ax_ticks = {'n-exp' : sorted(df_nb['n-exp'].unique()),
                'nb' : sorted(df_nb['nb'].unique()),
                'lr-ratio' : sorted(df_nb['lr-ratio'].unique())}

    pl = df_nb.pivot(values = 'mean_ns/q', index = y, columns = x)

    fig = plt.figure()
    k=0.5
    fig.set_figwidth(6*k)
    fig.set_figheight(4*k)

    minval = df_nb['mean_ns/q'].min()
    maxval = df_nb['mean_ns/q'].max()
    print(f"{minval=}   {maxval=}")
    #plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=.5, vmax=vmax))
    plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=minval, vmax=maxval), rasterized=True)
    #plt.colorbar()
    plt.xlabel(get_label(x, 'x'), fontsize=12)
    plt.ylabel(get_label(y, 'y'), fontsize=12)
    plt.xticks(range(0,26,5), fontsize=10)
    plt.xlim(0,27)
    plt.yticks(range(0,-26,-5), fontsize=10)
    
    # plt.yticks([i for i in range(5,26,2)])
    plt.title(get_title(title, x, y, col, plane))
    if saveFlag:
        plt.savefig(f"../plots/{filename}.pdf", dpi=300, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Run with arguments <csv_path> <#blocks> <title> <save_1_0>")
        exit()
    data_path = sys.argv[1]
    fname=Path(data_path).stem
    print(f"{fname=}")
    try:
        nb = int(sys.argv[2])
    except ValueError:
        nb = None
    title = sys.argv[3]
    saveFlag= int(sys.argv[4])
    print(f"ARGS:\n\t{data_path=}\n\t{nb=}\n\t{saveFlag}")
    df_3d = get3Ddata(data_path)
    nmin = int(df_3d['n-exp'].min())
    nmax = int(df_3d['n-exp'].max())
    #print(f"{nmin=}  {nmax=}")
    #if nexp > nmax or nexp < nmin:
    #    print(f"Error:\n\t'n-exp' must be between the file's range: {nmin}..{nmax}\n")
    #    exit()

    print("n:", sorted(df_3d['n-exp'].unique()))
    print("nb:", sorted(df_3d['nb'].unique()))
    print("lr:", sorted(df_3d['lr-ratio'].unique()))

    heat_map('n-exp', 'lr-ratio', nb, df_3d, title,fname,saveFlag, vmax=30)
