#!/bin/bash
if [ "$#" -ne 9 ]; then
    echo "Run as"
    printf "       ./run3Dplot    <bin-path>      <dev>    <n1> <n2>   <nb1> <nb2>   <lr1> <lr2>   <outfile_path>\n\n"
    printf "e.g:   ./run3Dplot.sh ../build/rtxrmq    0       16 26         1 12          1 15      ../data/results.csv\n"
    echo "note: values are exponents of 2^x"
    exit
fi
binPath=${1}
dev=${2}
n1=${3}
n2=${4}
nb1=${5}
nb2=${6}
lr1=${7}
lr2=${8}
outfile_path=${9}

printf "args:\nbinPath=${binPath}   dev=${dev}   n=${n1}-${n2}     nb=${nb1}-${nb2}     lr=${lr1}-${lr2}   outfile_path=${outfile_path}\n\n"
dev=0
#n=$((2**26))
q=$((2**26))
#lr=-1
nt=8
alg=5
reps=10
[ ! -f "results/data.csv" ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}

for(( n=$n1; n<=$n2; n++ ))
do
    for(( nb=$nb1; nb<=$nb2; nb++ ))
	do
        for(( lr=$lr1; lr<=$lr2; lr++ ))
		do
            narg=$((2**$n))
            echo "***EXECUTING***:${binPath} $narg $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time ${outfile_path}"
			${binPath} $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time ../results/3D_plot.csv
		done
	done
done

