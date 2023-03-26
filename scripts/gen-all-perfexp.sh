#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_perfexp.py

# TITAN RTX
echo "TITAN RTX Plots"
python ${SCRIPT} time 1 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'

python ${SCRIPT} speedup 1 TITANRTX ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA-GPU'
printf "\n"





# RTX 3090Ti
echo "RTX3090Ti Plots"
python ${SCRIPT} time 1 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'

python ${SCRIPT} speedup 1 RTX3090Ti ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA-GPU'
printf "\n"



# RTX 4090
echo "RTX4090 Plots"
python ${SCRIPT} time 1 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'

python ${SCRIPT} speedup 1 RTX4090 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-RTX4090-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA-GPU'
printf "\n"


# A100
echo "A100 Plots"
python ${SCRIPT} time 1 A100 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-A100-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-A100-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-A100-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-A100-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-A100-ALG7.csv 'LCA-GPU'

python ${SCRIPT} speedup 1 A100 ${CSVPATH}/perfexp-TRPRO-5975WX-ALG1.csv 'HRMQ'\
                                 ${CSVPATH}/perfexp-A100-ALG3.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-A100-ALG5.csv 'RTXRMQ-B (optimal)'\
                                 ${CSVPATH}/perfexp-A100-constNB-ALG5.csv 'RTXRMQ-B ($n_b=2^{9}$)'\
                                 ${CSVPATH}/perfexp-A100-constBS-ALG5.csv 'RTXRMQ-B ($B=2^{15}$)'\
                                 ${CSVPATH}/perfexp-A100-ALG7.csv 'LCA-GPU'
printf "\n"
