#!/bin/bash
CSVPATH=../csv-to-plot

# RTX 3090Ti
echo "RTX3090Ti Plots"
python plot_perf.py time 1 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python plot_perf.py speedup 1 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
printf "\n"



# RTX 4090
echo "RTX4090 Plots"
python plot_perf.py time 1 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

python plot_perf.py speedup 1 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'
printf "\n"


# A100
echo "A100 Plots"
python plot_perf.py time 1 A100 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-A100-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-A100-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-A100-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-A100-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-A100-ALG7.csv 'LCA-GPU'

python plot_perf.py speedup 1 A100 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-A100-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-A100-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perf-A100-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-A100-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-A100-ALG7.csv 'LCA-GPU'
printf "\n"
