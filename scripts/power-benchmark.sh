#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <seed> <reps>    <n>          <q>         <nb>    <lr> <filename>\n\n"
    printf "e.g: ${0}  0     8     5    1321    16  \$((2**20))  \$((2**26))  \$((2**9))   -2   RTX4090\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - 'nb' (number of blocks) will only be considered for algorithms 5 and 8\n"
    printf "  - prefix (power) and suffix (alg) will be added to filename\n"
    printf "  - 'lr' > 0 is the actual query range size, i.e., lr = (r-l)+1\n"
    printf "  - 'lr' < 0 the distributions (-1 large,-2 medium,-3 small)\n\n"
    printf "  - NOTE: check first that the result is correct (--check param)\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
seed=${4}
reps=${5}
n=${6}
q=${7}
nb=${8}
lr=${9}
outfile_path=../data/power-${10}-n${n}-q${q}-lr${lr}-r${reps}-s${seed}-ALG${alg}.csv
binary=./rtxrmq

printf "args:\ndev=${dev} nt=${nt} alg=${alg} rea=${rea} reps=${reps} n=${n} q=${q} nb=${nb} lr=${lr} outfile_path=${outfile_path}\n\n"
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATE}"

# change to bin directory
cd ../build

printf "${binary} $n $q ${lr} ${alg} --nb ${nb} --reps $reps --nt $nt --dev $dev --save-power=${outfile_path} --seed ${seed}\n"
        ${binary} $n $q ${lr} ${alg} --nb ${nb} --reps $reps --nt $nt --dev $dev --save-power=${outfile_path} --seed ${seed}
# come back to scripts directory
cd ../scripts
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "FINISH #DATE = ${DATE}"
