import pandas as pd
import numpy as np


# x : nb = n/bs
# y : lr-ratio = n/lr
# z : n
# v : ns/q

df_3d = pd.read_csv("../results/3D_plot.csv")
# df_3d.head()

hc = df_3d
hc = hc[hc['lr'] > 0]


hc['n-exp'] = np.log2(hc['n'])
hc['nb'] = np.log2(hc['n'] / hc['bs'])
hc['lr-ratio'] = np.log2(hc['lr'] / hc['n'])
display(hc.head())
hc[['n-exp', 'nb', 'lr-ratio', 'ns/q']].to_csv("heat_cube.csv", header=False, index=False)
