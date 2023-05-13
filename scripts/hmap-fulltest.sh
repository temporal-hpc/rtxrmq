#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <testname>\n\n"
    printf "e.g: ${0}  0      8    5    RTX3090Ti\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
testname=${4}
printf "dev=${0}  nt=${nt}  alg=${alg}  testname=${testname}"
# small n, many rea/reps
#./hmap-benchmark.sh ${dev} ${nt} ${alg}  16  16     0 12   26   0 26  0 26  ${testname}

# medium n, intermediate number of rea/reps
#./hmap-benchmark.sh ${dev} ${nt} ${alg}  8    8    13 19   26   0 26  0 26  ${testname}

# large n, small number of rea/reps
#./hmap-benchmark.sh ${dev} ${nt} ${alg}  4    4    20 26   26   0 26  0 26  ${testname}
./hmap-benchmark.sh ${dev} ${nt} ${alg}  2    2    25 26   26   0 26  0 26  ${testname}
