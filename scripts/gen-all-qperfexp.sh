#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_qperfexp.py
echo "TIME plots"


# ******************** large (l,r) range *********************
python ${SCRIPT} -1 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ-CPU'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG7.csv 'LCA'

python ${SCRIPT} -1 1 1e-2 1e5 RTX6000ADA ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ-CPU'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG7.csv 'LCA'



# ******************** medium (l,r) range ********************
python ${SCRIPT} -2 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ-CPU'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG7.csv 'LCA'

python ${SCRIPT} -2 1 1e-2 1e5 RTX6000ADA ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192-cores'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG7.csv 'LCA'



# ********************* small (l,r) range ********************
python ${SCRIPT} -3 1 1e-2 1e5 RTX3090Ti ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192-cores'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX3090Ti-ALG7.csv 'LCA'

python ${SCRIPT} -3 1 1e-2 1e5 RTX6000ADA ${CSVPATH}/qexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192-cores'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/qexp-RTX6000ADA-ALG7.csv 'LCA'