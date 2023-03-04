#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <testname>\n\n"
    printf "e.g: ${0}  0      8   RTX3090Ti\n\n"
    exit
fi
dev=${1}
nt=${2}
testname=${3}
printf "dev=${0}  nt=${nt}  testname=${testname}"
# small n, many rea/reps
./hmap-benchmark.sh ${dev} ${nt} 5  16  16     0 12   26   0 24  0 24  ${testname}

# medium n, intermediate number of rea/reps
./hmap-benchmark.sh ${dev} ${nt} 5  8    8    13 19   26   0 24  0 24  ${testname}

# large n, small number of rea/reps
./hmap-benchmark.sh ${dev} ${nt} 5  4    4    20 24   26   0 24  0 24  ${testname}
