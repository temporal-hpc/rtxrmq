#!/bin/bash

# RTX 3090Ti
echo "RTX3090Ti Plots"
python plot_perf.py time 1 RTX3090Ti ../csv-finales/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ../csv-finales/perf-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ../csv-finales/perf-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ../csv-finales/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{15}$)'\
                                 ../csv-finales/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($B=2^9$)'\
                                 ../csv-finales/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python plot_perf.py speedup 1 RTX3090Ti ../csv-finales/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ../csv-finales/perf-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ../csv-finales/perf-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ../csv-finales/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{15}$)'\
                                 ../csv-finales/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($B=2^9$)'\
                                 ../csv-finales/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
printf "\n"



# RTX 4090
echo "RTX4090 Plots"
python plot_perf.py time 1 RTX4090 ../csv-finales/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ../csv-finales/perf-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ../csv-finales/perf-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ../csv-finales/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{15}$)'\
                                 ../csv-finales/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($B=2^9$)'\
                                 ../csv-finales/perf-RTX4090-ALG7.csv 'LCA-GPU'

python plot_perf.py speedup 1 RTX4090 ../csv-finales/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ../csv-finales/perf-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ../csv-finales/perf-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ../csv-finales/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{15}$)'\
                                 ../csv-finales/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($B=2^9$)'\
                                 ../csv-finales/perf-RTX4090-ALG7.csv 'LCA-GPU'
printf "\n"
