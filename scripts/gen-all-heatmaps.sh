#!/bin/bash

# HRMQ (CPU)
python plot_heat_map.py ../csv-finales/hmap-THREADRIPPER-5975WX-ALG1.csv None "HRMQ-CPU (32-core 5975WX)" 1

# TITAN RTX
python plot_heat_map.py ../csv-finales/hmap-TITANRTX-ALG3.csv None "RTXRMQ (TITAN RTX)" 1
python plot_heat_map.py ../csv-finales/hmap-TITANRTX-ALG5.csv 15 "RTXRMQ-B (TITAN RTX)" 1
python plot_heat_map.py ../csv-finales/hmap-TITANRTX-ALG7.csv None "LCA-GPU (TITAN RTX)" 1

# RTX 3090Ti
python plot_heat_map.py ../csv-finales/hmap-RTX3090Ti-ALG3.csv None "RTXRMQ (RTX 3090Ti)" 1
python plot_heat_map.py ../csv-finales/hmap-RTX3090Ti-ALG5.csv 15 "RTXRMQ-B (RTX 3090Ti)" 1
python plot_heat_map.py ../csv-finales/hmap-RTX3090Ti-ALG7.csv None "LCA-GPU (RTX 3090Ti)" 1

# RTX 4090
python plot_heat_map.py ../csv-finales/hmap-RTX4090-ALG3.csv None "RTXRMQ" 1
python plot_heat_map.py ../csv-finales/hmap-RTX4090-ALG5.csv 15 "RTXRMQ-B" 1
python plot_heat_map.py ../csv-finales/hmap-RTX4090-ALG7.csv None "LCA-GPU" 1
python plot_best_heatmap_from_3D.py ../csv-finales/hmap-RTX4090-ALG5.csv "RTXRMQ-B, Best Configuration" 1

# A100
python plot_heat_map.py ../csv-finales/hmap-A100-ALG3.csv None "RTXRMQ (A100)" 1
python plot_heat_map.py ../csv-finales/hmap-A100-ALG5.csv 15 "RTXRMQ-B (A100)" 1
python plot_heat_map.py ../csv-finales/hmap-A100-ALG7.csv None "LCA-GPU (A100)" 1
