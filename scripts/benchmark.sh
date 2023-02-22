#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "Run as"
    printf "     ${0}      <bin-path>  <alg> <dev>  <n1> <n2>  <nb1> <nb2>  <lr1> <lr2>   <filename>\n\n"
    printf "e.g: ${0} ../build/rtxrmq    5     0     16   26      1   12       1   15     results-RTX3090Ti\n"
    printf "note:\n  - the *.csv extension will be placed automatically\n  - n,nb,lr values are exponents of 2^x\n\n"
    exit
fi
binPath=${1}
alg=${2}
dev=${3}
n1=${4}
n2=${5}
nb1=${6}
nb2=${7}
lr1=${8}
lr2=${9}
outfile_path=../data/${10}-ALG${alg}.csv

printf "args:\nbinPath=${binPath}  alg=${alg}  dev=${dev}   n=${n1}-${n2}     nb=${nb1}-${nb2}     lr=${lr1}-${lr2}   outfile_path=${outfile_path}\n\n"
dev=0
#n=$((2**26))
q=$((2**20))
#lr=-1
nt=8
reps=10
[ ! -f "results/data.csv" ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}

for(( n=$n1; n<=$n2; n++ ))
do
    for(( nb=$nb1; nb<=$nb2; nb++ ))
    do
        for(( lr=$lr1; lr<=$lr2; lr++ ))
        do
            narg=$((2**$n))
            printf "\n\n\n\n\n\n\n"
            printf "***EXECUTING***:\n\t${binPath} $narg $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time ${outfile_path}\n"
            ${binPath} $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time ${outfile_path}
        done
    done
done
