#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_qperfexp.py

# TIME
echo "TIME plots"
# **** large (l,r) range ****
#python ${SCRIPT} -1 1 1e-2 1e5 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-QPERF-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -1 1 1e-2 1e5 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** medium (l,r) range ****
#python ${SCRIPT} -2 1 1e-2 1e5 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-QPERF-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -2 1 1e-2 1e5 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** small (l,r) range ****
#python ${SCRIPT} -3 1 1e-2 1e5 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-QPERF-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-QPERF-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -3 1 1e-2 1e5 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'
