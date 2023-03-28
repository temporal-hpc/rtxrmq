#!/bin/bash
printf "ARGS ${#}\n"
if [ "$#" -ne 15 ]; then
    echo "Run as"
    printf "    ${0} <dev> <nt> <alg> <rea> <reps>  <n1>         <n2>        <dn>       <q1>        <q2>         <dq>   <bs|nb> <block>     <lr> <name>\n\n"
    printf "e.g ${0}  0     8     5     8     10  \$((10**6)) \$((10**8))  \$((10**6)) \$((2**26)) \$((2**26))   100      bs    \$((2**15))  -1  RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix (alg) will be added to filename\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
rea=${4}
reps=${5}
n1=${6}
n2=${7}
dn=${8}
q1=${9}
q2=${10}
dq=${11}
bs_or_nb=${12}
bsize=${13}
lr=${14}
outfile_path=../data/perf-${15}-ALG${alg}.csv
binary=./rtxrmq

printf "args:\ndev=${dev} nt=${nt} alg=${alg} rea=${rea} reps=${reps} n=${n1}-${n2} (dn=${dn}) q=${q1}-${q2} (dq=${dq}) bs|nb=${bsize} lr=${lr}   outfile_path=${outfile_path}\n\n"
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

# change to bin directory
cd ../build

for(( n=$n1; n<=$n2; n+=${dn} ))
do
    for(( q=$q1; q<=$q2; q+=${dq} ))
    do
        for(( R=1; R<=$rea; R++ ))
        do
            printf "\n\n\n\n\n\n\n\n"
            SEED=${RANDOM}
            printf "REALIZATION $R -> n=$n q=$q\n"
	    if [ "${bs_or_nb}" = "bs" ]; then
		    printf "${binary} $n $q ${lr} ${alg} --bs ${bsize} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		            ${binary} $n $q ${lr} ${alg} --bs ${bsize} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}
        else
		    printf "${binary} $n $q ${lr} ${alg} --nb ${bsize} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		            ${binary} $n $q ${lr} ${alg} --nb ${bsize} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}
	    fi
        done
    done
done
# come back to scripts directory
cd ../scripts
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
