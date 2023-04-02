#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_perf.py

# TIME
echo "TIME plots"
# **** large (l,r) range ****
python ${SCRIPT} -1 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'


# **** medium (l,r) range ****
python ${SCRIPT} -2 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'


# **** small (l,r) range ****
python ${SCRIPT} -3 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'




# SPEEDUP
echo "SPEEDUP plots"
# **** large (l,r) range ****
python ${SCRIPT} -1 speedup 1 0 25 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 speedup 1 0 55 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 speedup 1 0 60 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'


# **** medium (l,r) range ****
python ${SCRIPT} -2 speedup 1 0 15 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 speedup 1 0 30 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 speedup 1 0 35 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'


# **** small (l,r) range ****
python ${SCRIPT} -3 speedup 1 0 5 TITANRTX ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 speedup 1 0 10 RTX3090Ti ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 speedup 1 0 25 RTX4090 ${CSVPATH}/perf-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perf-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'
