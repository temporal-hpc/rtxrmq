#!/bin/bash
CSVPATH=../csv-to-plot
SCRIPT=plot_power.py

# power
echo "*** POWER & SCALING ***"
# **** large (l,r) range ****
python ${SCRIPT} 1 -1 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-1-r512-s1321-ALG1.csv "\$\\mathrm{REF}_{\\mathrm{CPU}}\$@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-1-r1-s1321-ALG2.csv 'Exhaustive' 1

## **** medium (l,r) range ****
python ${SCRIPT} 1 -2 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-2-r512-s1321-ALG1.csv "\$\\mathrm{REF}_{\\mathrm{CPU}}\$@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-2-r1-s1321-ALG2.csv 'Exhaustive' 1

## **** small (l,r) range ****
python ${SCRIPT} 1 -3 0 700 'RTX6000ADA' ${CSVPATH}/power-2X-EPYC9654-96C-n100000000-q67108864-lr-3-r512-s1321-ALG1.csv "\$\\mathrm{REF}_{\\mathrm{CPU}}\$@192c" 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG5.csv 'RTXRMQ' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG7.csv 'LCA' 512\
                                         ${CSVPATH}/power-RTX6000ADA-n100000000-q67108864-lr-3-r512-s1321-ALG2.csv 'Exhaustive' 512

printf "Generating RMQs per Joule......."
python plot_rmq_per_joule.py
printf "done\n"


printf "Generating Scaling Arch........."
python plot_scaling_arch.py
printf "done\n"


printf "Generating Scaling SMs.........."
python plot_scaling_sm.py
printf "done\n\n"
