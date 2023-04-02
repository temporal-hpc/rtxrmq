#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_perfexp.py

# TIME
echo "TIME plots"
# **** large (l,r) range ****
python ${SCRIPT} -1 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** medium (l,r) range ****
python ${SCRIPT} -2 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** small (l,r) range ****
python ${SCRIPT} -3 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'




# SPEEDUP
echo "SPEEDUP plots"
# **** large (l,r) range ****
python ${SCRIPT} -1 speedup 1 0 200 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 speedup 1 0 200 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 speedup 1 0 200 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** medium (l,r) range ****
python ${SCRIPT} -2 speedup 1 0 150 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 speedup 1 0 150 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 speedup 1 0 150 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'


# **** small (l,r) range ****
python ${SCRIPT} -3 speedup 1 0 200 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 speedup 1 0 200 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 speedup 1 0 200 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'
