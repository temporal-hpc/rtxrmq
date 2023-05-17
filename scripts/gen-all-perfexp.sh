#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_perfexp.py

# TIME
echo "TIME plots"
# **** large (l,r) range ****
#python ${SCRIPT} -1 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -1 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -1 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

python ${SCRIPT} -1 time 1 1e-2 1e6 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'


# **** medium (l,r) range ****
#python ${SCRIPT} -2 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -2 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -2 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

python ${SCRIPT} -2 time 1 1e-2 1e3 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'


# **** small (l,r) range ****
#python ${SCRIPT} -3 time 1 1e-2 1e2 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -3 time 1 1e-2 1e2 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -3 time 1 1e-2 1e2 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

python ${SCRIPT} -3 time 1 1e-2 1e2 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'



# SPEEDUP
#echo "SPEEDUP plots"
# **** large (l,r) range ****
#python ${SCRIPT} -1 speedup 1 0 200 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -1 speedup 1 0 200 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -1 speedup 1 0 200 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

#python ${SCRIPT} -1 speedup 1 0 20 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'


# **** medium (l,r) range ****
#python ${SCRIPT} -2 speedup 1 0 150 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -2 speedup 1 0 150 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -2 speedup 1 0 150 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

#python ${SCRIPT} -2 speedup 1 0 20 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'

# **** small (l,r) range ****
#python ${SCRIPT} -3 speedup 1 0 200 TITANRTX ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-TITANRTX-ALG7.csv 'LCA'
#
#python ${SCRIPT} -3 speedup 1 0 200 RTX3090Ti ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX3090Ti-ALG7.csv 'LCA'
#
#python ${SCRIPT} -3 speedup 1 0 200 RTX4090 ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX4090-ALG7.csv 'LCA'

#python ${SCRIPT} -3 speedup 1 0 20 RTX6000ADA ${CSVPATH}/perfexp-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG7.csv 'LCA'\
#                                 ${CSVPATH}/perfexp-RTX6000ADA-ALG2.csv 'Exhaustive'
