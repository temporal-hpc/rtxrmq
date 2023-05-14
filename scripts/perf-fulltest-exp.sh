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
name="perfexp-${4}"

const_bs=15
const_nb=9

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

printf "args dev=${dev} nt=${nt} alg=${alg} bsize=${bsize} name=${name}\n\n"

if [ "$alg" -ne 5 ] && [ "$alg" -ne 8 ]; then
	for lr in {-1..-3}
	do
		if [ "$alg" -eq 2 ]; then
			#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   18     26   26       nb       0      ${lr}  ${name}
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}   2      2      19   26     26   26       nb       0      ${lr}  ${name}
        elif [ "$alg" -ne 3 ]; then
			#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb       0      ${lr}  ${name}
		else
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   24      26   26       nb       0      ${lr}  ${name}
		fi
	done
elif [ "$alg" -eq 5 ]; then
	# UNIFORM DISTRIBUTION (large values)
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   21     26   26       nb        14      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   26     26   26       nb         9      -1   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -1   ${name}-constBS
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -1   ${name}-constNB


	# LOGNORMAL DISTRIBUTION (medium values) EXP 0.6
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   22     26   26       nb         1      -2   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      23   26     26   26       nb         9      -2   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -2   ${name}-constBS
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -2   ${name}-constNB


	# LOGNORMAL DISTRIBUTION (small values) EXP 0.3
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   20     26   26       nb        11      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      21   21     26   26       nb        12      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   26     26   26       nb        13      -3   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -3   ${name}-constBS
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       nb   ${const_nb}  -3   ${name}-constNB

elif [ "$alg" -eq 8 ]; then
	# UNIFORM DISTRIBUTION (large values)
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   21     26   26       bs        9      -1   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   26     26   26       bs       18      -1   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -1   ${name}-constBS

	# LOGNORMAL DISTRIBUTION (medium values) EXP 0.6
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   22     26   26       bs        20      -2   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      23   26     26   26       bs        18      -2   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -2   ${name}-constBS

	# LOGNORMAL DISTRIBUTION (small values) EXP 0.3
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   21     26   26       bs        10      -3   ${name}
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      22   26     26   26       bs        18      -3   ${name}
	# constant BS and NB
	#./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32       0   26     26   26       bs   ${const_bs}  -3   ${name}-constBS

fi
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL BENCHMARK EXP FINISHED: args dev=${dev} nt=${nt} alg=${alg} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
