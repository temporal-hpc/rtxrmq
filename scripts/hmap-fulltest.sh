#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Run as"
    printf "     ${0} <dev> <testname>\n\n"
    printf "e.g: ${0}  0     RTX3090Ti\n\n"
    exit
fi
dev=${1}
testname=${2}
printf "dev=${0}  testname=${testname}"
# small n, many rea/reps
./hmap-benchmark.sh ${dev} 8 5  16  16    10 14   26   0 24  0 24  ${testname}

# medium n, intermediate number of rea/reps
./hmap-benchmark.sh ${dev} 8 5  8    8    15 19   26   0 24  0 24  ${testname}

# large n, small number of rea/reps
./hmap-benchmark.sh ${dev} 8 5  4    4    20 24   26   0 24  0 24  ${testname}
