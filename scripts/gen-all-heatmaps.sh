#!/bin/bash
PATH_CSV=../csv-to-plot

# HRMQ (CPU)
python plot_heat_map.py $PATH_CSV/hmap-TRPRO-5975WX-ALG1.csv    None "HRMQ-CPU" 1
python plot_heat_map.py $PATH_CSV/hmap-2X-EPYC9654-96C-ALG1.csv None "HRMQ-CPU" 1

# RTX 6000 ADA
python plot_heat_map.py             $PATH_CSV/hmap-RTX6000ADA-ALG2.csv None "Exhaustive" 1
python plot_best_heatmap_from_3D.py $PATH_CSV/hmap-RTX6000ADA-ALG5.csv "RTXRMQ" 1
python plot_heat_map.py             $PATH_CSV/hmap-RTX6000ADA-ALG7.csv None "LCA" 1
