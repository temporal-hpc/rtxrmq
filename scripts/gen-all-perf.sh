#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_perf.py

# TIME
#echo "TIME plots"
# **** large (l,r) range ****
#python ${SCRIPT} -1 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -1 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -1 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -1 time 1 1e-1 1e1 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA-GPU'
#echo ""



# **** medium (l,r) range ****
#python ${SCRIPT} -2 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -2 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -2 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -2 time 1 1e-1 1e1 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA-GPU'
#echo ""





# **** small (l,r) range ****
#python ${SCRIPT} -3 time 1 1e-1 1e2 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -3 time 1 1e-1 1e2 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -3 time 1 1e-1 1e2 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

#python ${SCRIPT} -3 time 1 1e-1 1e1 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA'\
#                                 ${CSVPATH}/perf-RTX6000ADA-ALG2.csv 'Exhaustive'
#echo ""






# SPEEDUP
echo "SPEEDUP plots"
# **** large (l,r) range ****
#python ${SCRIPT} -1 speedup 1 0 25 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -1 speedup 1 0 55 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -1 speedup 1 0 60 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -1 speedup 1 -0.2 15 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c@192-cores'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG2.csv 'Exhaustive'
echo ""





# **** medium (l,r) range ****
#python ${SCRIPT} -2 speedup 1 0 15 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -2 speedup 1 0 30 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -2 speedup 1 0 35 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -2 speedup 1 -0.2 9 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c@192-cores'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG2.csv 'Exhaustive'
echo ""







# **** small (l,r) range ****
#python ${SCRIPT} -3 speedup 1 0 5 TITANRTX ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-TITANRTX-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -3 speedup 1 0 10 RTX3090Ti ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX3090Ti-ALG7.csv 'LCA-GPU'
#
#python ${SCRIPT} -3 speedup 1 0 25 RTX4090 ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
#                                 ${CSVPATH}/perf-RTX4090-ALG5.csv 'RTXRMQ'\
#                                 ${CSVPATH}/perf-RTX4090-ALG7.csv 'LCA-GPU'

python ${SCRIPT} -3 speedup 1 0 6 RTX6000ADA ${CSVPATH}/perf-2X-EPYC9654-96C-ALG1.csv 'HRMQ@192c'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG5.csv 'RTXRMQ'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG7.csv 'LCA'\
                                 ${CSVPATH}/perf-RTX6000ADA-ALG2.csv 'Exhaustive'
echo ""
