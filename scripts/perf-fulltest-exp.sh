#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <filename>\n"
    printf "e.g: ${0}   0     8    5    RTX4090\n"
    printf "\nnote:\n"
    printf "  - dev   : GPU device ID\n"
    printf "  - nt    : number of CPU threads (relevant for CPU methods)\n"
    printf "  - alg   : algorithm (check ./rtxrmq help message)\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
name=${4}

const_bs=15
const_nb=9
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

printf "args dev=${dev} nt=${nt} alg=${alg} bsize=${bsize} name=${name}\n\n"

if [ "$alg" -ne 5 ]; then
	for lr in {-1..-3}
	do
		if [ "$alg" -ne 3 ]; then
			#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb       0      ${lr}  ${name}
		else
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   24      26   26       nb       0      ${lr}  ${name}
		fi
	done
else
	# UNIFORM DISTRIBUTION (large values)
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   12     26   26       nb         0      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      13   13     26   26       nb        13      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      14   16     26   26       nb         9      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      17   17     26   26       nb        10      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      18   18     26   26       nb        12      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      19   21     26   26       nb        14      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   24     26   26       nb         9      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      25   26     26   26       nb         8      -1   ${name}
	# constant BS and NB
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -1   ${name}-constBS
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -1   ${name}-constNB





	# LOGNORMAL DISTRIBUTION (medium values) EXP 0.6
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   22     26   26       nb         1      -2   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      23   26     26   26       nb         9      -2   ${name}
	# constant BS and NB
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -2   ${name}-constBS
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -2   ${name}-constNB




	# LOGNORMAL DISTRIBUTION (small values) EXP 0.3
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0    9     26   26       nb         0      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      10   11     26   26       nb         1      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      12   12     26   26       nb         7      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      13   13     26   26       nb         7      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      14   14     26   26       nb         8      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      15   15     26   26       nb         9      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      16   18     26   26       nb        10      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      19   20     26   26       nb        11      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      21   21     26   26       nb        12      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   24     26   26       nb        13      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      25   26     26   26       nb        14      -3   ${name}
	# constant BS and NB
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -3   ${name}-constBS
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -3   ${name}-constNB





	# LOGNORMAL V2
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   24      26   26       nb        1      -4    ${name}
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      25   26      26   26       nb        7      -4    ${name}

	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0    9      26   26       nb        0      -5    ${name}
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      10   19      26   26       nb       11      -5    ${name}
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      20   24      26   26       nb       11      -5    ${name}
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      25   26      26   26       nb       13      -5    ${name}
	## constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -4   ${name}-constBS
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -5   ${name}-constBS
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -4   ${name}-constNB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -5   ${name}-constNB
fi
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL BENCHMARK EXP FINISHED: args dev=${dev} nt=${nt} alg=${alg} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
