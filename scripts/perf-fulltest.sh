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

const_bs=$((2**15))
const_nb=$((2**9))
N1=$((10**6))
N2=$((10**8))
N2A3=$((16*(10**6)))
DN=$((10**6))

Q1=$((2**26))
Q2=$((2**26))
DQ=$((100))

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

printf "args dev=${dev} nt=${nt} alg=${alg} name=${name}, N1=${N1}, N2=${N2}, N2A3=${N2A3}, DN=${DN}, Q1=${Q1}, Q2=${Q2}, DQ=${DQ}\n\n"

if [ "$alg" -ne 5 ]; then
	for lr in {-1..-3}
	do
		if [ "$alg" -ne 3 ]; then
			#./perf-benchmark.sh     <dev>  <nt>  <alg> <rea> <reps>  <n1>   <n2>   <dn>      <q1>  <q2>   <dq> <bs|nb> <block> <lr>   <name>
			./perf-benchmark.sh     ${dev} ${nt} ${alg}  16    16    ${N1} ${N2}   ${DN}       ${Q1} ${Q2}  ${DQ}   nb       1   ${lr}  ${name}
		else
			./perf-benchmark.sh     ${dev} ${nt} ${alg}  16    16    ${N1} ${N2A3} ${DN}       ${Q1} ${Q2}  ${DQ}   nb       1   ${lr}  ${name}
		fi
	done
else
	# LR=-1  UNIFORM DISTRIBUTION (large values)
	#./perf-benchmark.sh     <dev>  <nt>  <alg> <rea> <reps>       <n1>         <n2>          <dn>    <q1>   <q2>   <dq>   <bs|nb> <block>    <lr>   <name>
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $(( 1*${N1}))  $(( 3*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb  $((2**14))  -1   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $(( 4*${N1}))  $((33*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb  $((2**9))   -1   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $((34*${N1}))  $((67*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb  $((2**8))   -1   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $((68*${N1})) $((100*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb  $((2**8))   -1   ${name}
	# constant BS and NB
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     bs   ${const_bs}  -1   ${name}-constBS
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     nb   ${const_nb}  -1   ${name}-constNB

	# LR=-2  LOGNORMAL DISTRIBUTION (medium values)
	#./perf-benchmark.sh     <dev>  <nt>  <alg> <rea> <reps>       <n1>         <n2>          <dn>    <q1>   <q2>   <dq>   <bs|nb>  <block>    <lr>   <name>
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16   $(( 1*${N1}))  $(( 6*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**1))   -2   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16   $(( 7*${N1}))  $((100*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**9))   -2   ${name}
	# constant BS and NB
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     bs   ${const_bs}  -2   ${name}-constBS
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     nb   ${const_nb}  -2   ${name}-constNB

	# LR=-3  LOGNORMAL DISTRIBUTION (small values)
	#./perf-benchmark.sh     <dev>  <nt>  <alg> <rea> <reps>       <n1>         <n2>          <dn>    <q1>   <q2>   <dq>   <bs|nb> <block>      <lr>   <name>
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $(( 1*${N1}))  $(( 1*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**11))   -3   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $(( 2*${N1}))  $(( 3*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**12))   -3   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $(( 4*${N1}))  $((33*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**13))   -3   ${name}
    ./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16    $((34*${N1}))  $((100*${N1}))    ${DN}   ${Q1} ${Q2}   ${DQ}     nb   $((2**14))   -3   ${name}
	# constant BS and NB
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     bs   ${const_bs}  -3   ${name}-constBS
	./perf-benchmark.sh      ${dev} ${nt} ${alg}  16    16        ${N1}         ${N2}         ${DN}   ${Q1} ${Q2}   ${DQ}     nb   ${const_nb}  -3   ${name}-constNB

fi
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL BENCHMARK FINISHED: args dev=${dev} nt=${nt} alg=${alg} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
