#!/bin/bash
PATH_CSV=../csv-to-plot

printf "*** HEATMAPS ***\n"
# HRMQ (CPU)
printf "Generating CPU heatmap for: hmap-TRPRO-5975WX-ALG1.csv.........."
python plot_heat_map.py $PATH_CSV/hmap-TRPRO-5975WX-ALG1.csv    None "HRMQ" 1
printf "done\n"

printf "Generating CPU heatmap for: hmap-2X-EPYC9654-96C-ALG1.csv......."
python plot_heat_map.py $PATH_CSV/hmap-2X-EPYC9654-96C-ALG1.csv None "HRMQ" 1
printf "done\n"

# RTX 6000 ADA
printf "Generating GPU heatmap for: hmap-RTX6000ADA-ALG2.csv............"
python plot_heat_map.py             $PATH_CSV/hmap-RTX6000ADA-ALG2.csv None "Exhaustive" 1
printf "done\n"
printf "Generating GPU heatmap for: hmap-RTX6000ADA-ALG5.csv............"
python plot_best_heatmap_from_3D.py $PATH_CSV/hmap-RTX6000ADA-ALG5.csv "RTXRMQ" 1
printf "done\n"
printf "Generating GPU heatmap for: hmap-RTX6000ADA-ALG7.csv............"
python plot_heat_map.py             $PATH_CSV/hmap-RTX6000ADA-ALG7.csv None "LCA" 1
printf "done\n\n"
