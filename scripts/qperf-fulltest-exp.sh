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
name=qexp-${4}
N1=27
N2=27
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"
printf "args dev=${dev} nt=${nt} alg=${alg} bsize=${bsize} name=${name}\n\n"
if [ "$alg" -ne 5 ] && [ "$alg" -ne 8 ] && [ "$alg" -ne 10 ]; then
	for lr in {-1..-3}
	do
			#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>   <n1>  <n2>  <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
			./perf-benchmark-exp.sh ${dev} ${nt} ${alg}    4     4    ${N1} ${N2}   0   26       nb       0      ${lr}  ${name}
	done
elif [ "$alg" -eq 5 ] || [ "$alg" -eq 10 ]; then
	# UNIFORM DISTRIBUTION (large values)
	#./perf-benchmark-exp.sh <dev> <nt>  <alg>  <rea> <reps>     <n1>   <n2>   <q1> <q2>  <bs-or-nb> <bsize>   <lr> <filename>
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       nb         9      -1   ${name}

	# LOGNORMAL DISTRIBUTION (medium values) EXP 0.6
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       nb         1      -2   ${name}

	# LOGNORMAL DISTRIBUTION (small values) EXP 0.3
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       nb        13      -3   ${name}
elif [ "$alg" -eq 8 ]; then
	# UNIFORM DISTRIBUTION (large values)
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       bs        18      -1   ${name}

	# LOGNORMAL DISTRIBUTION (medium values) EXP 0.6
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       bs        18      -2   ${name}

	# LOGNORMAL DISTRIBUTION (small values) EXP 0.3
	./perf-benchmark-exp.sh ${dev} ${nt} ${alg}  16     32      ${N1}  ${N2}     0   26       bs        13      -3   ${name}
fi
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL BENCHMARK EXP FINISHED: args dev=${dev} nt=${nt} alg=${alg} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
