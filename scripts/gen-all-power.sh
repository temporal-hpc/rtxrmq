#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_power.py

# power
echo "Power plots"
# **** large (l,r) range ****
python ${SCRIPT} 1 -1 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-1-r512-s1321-ALG1.csv "HRMQ@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r1-s1321-ALG2.csv 'Exhaustive' 1
#echo ""
#
## **** medium (l,r) range ****
python ${SCRIPT} 1 -2 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-2-r512-s1321-ALG1.csv "HRMQ@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r1-s1321-ALG2.csv 'Exhaustive' 1
echo ""

## **** small (l,r) range ****
python ${SCRIPT} 1 -3 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-3-r512-s1321-ALG1.csv "HRMQ@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG2.csv 'Exhaustive' 512
echo ""
