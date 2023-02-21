dev=0
nt=8
bs=$((2**15))
reps=10

[ -e "results/data.csv" ] || echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > results/data.csv

cd ../build/
for alg in {1,3}
do
	for n in {16..26..2}
	do
		for q in {10..25..3}
		do
			for lr in {5..25..4}
			do
				if [ $lr -lt $n ]
				then
					./rtxrmq $((2**$n)) $((2**$q)) $((2**$lr)) $alg --bs $bs --nt $nt --reps $reps --dev $dev
				fi
			done
			./rtxrmq $((2**$n)) $((2**$q)) -1 $alg --bs $bs --nt $nt --reps $reps --dev $dev
		done
	done
done

