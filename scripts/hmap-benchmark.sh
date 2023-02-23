#!/bin/bash
if [ "$#" -ne 13 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <rea> <reps>   <n1> <n2>  <q>  <nb1> <nb2>   <lr1> <lr2>   <filename>\n\n"
    printf "e.g: ${0}  0     8     5     8     10     16   26   26      1   12        1   15     results-RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix (alg) will be added to filename\n"
    printf "  - n,q,bsize,lr values are exponents of 2^x\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
rea=${4}
reps=${5}
n1=${6}
n2=${7}
q=$((2**${8}))
nb1=${9}
nb2=${10}
if [ "$alg" -le 4 ]; then
    nb2=${nb1}
fi
lr1=${11}
lr2=${12}
outfile_path=../data/hmap-${13}-ALG${alg}.csv
binary=./rtxrmq

printf "args:\nalg=${alg}  dev=${dev}  nt=${nt} rea=${rea} reps=${reps}  n=${n1}-${n2} q=${q}  nb=${nb1}-${nb2}     lr=${lr1}-${lr2}   outfile_path=${outfile_path}\n\n"
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
                printf "REALIZATION $R -> n=$((2**$n)) nb=$((2**$nb)) lr=$((2**$lr))\n"
                printf "${binary} $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
                        ${binary} $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}
            done
        done
    done
done
# come back to scripts directory
cd ../scripts
