#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "Run as"
    printf "     ${0} <dev> <alg> <rea> <n1> <n2>  <nb1> <nb2>  <lr1> <lr2>   <filename>\n\n"
    printf "e.g: ${0}  0     5      9    16   26      1   12       1   15     results-RTX3090Ti\n"
    printf "note:\n  - the *.csv extension will be placed automatically\n  - n,nb,lr values are exponents of 2^x\n\n"
    exit
fi
dev=${1}
alg=${2}
rea=${3}
n1=${4}
n2=${5}
nb1=${6}
nb2=${7}
lr1=${8}
lr2=${9}
outfile_path=../data/time-${10}-ALG${alg}.csv
outfile_path_power=../data/power-${10}-ALG${alg}.csv
binary=./rtxrmq

printf "args:\nalg=${alg}  dev=${dev}   n=${n1}-${n2}     nb=${nb1}-${nb2}     lr=${lr1}-${lr2}   outfile_path=${outfile_path}\n\n"
dev=0
#n=$((2**26))
q=$((2**20))
#lr=-1
nt=8
reps=10
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
#DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
#echo "#DATE = ${DATE}" >> ${outfile_path}

# change to bin directory
cd ../build

for(( n=$n1; n<=$n2; n++ ))
do
    for(( nb=$nb1; nb<=$nb2; nb++ ))
    do
        for(( lr=$lr1; lr<=$lr2; lr++ ))
        do
            for(( R=1; R<=$rea; R++ ))
            do
                printf "\n\n\n\n\n\n\n\n"
                SEED=${RANDOM}
                printf "REALIZATION $R\n:\n./rtxrmq $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --save-power=${outfile_path_power} --seed ${SEED}\n"
                ${binary} $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --save-power=${outfile_path_power} --seed ${SEED}
            done
        done
    done
done
# come back to scripts directory
cd ../scripts
