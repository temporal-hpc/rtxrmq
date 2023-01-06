dev=0
nt=8
bs=1024

[ -e "resuslts/data.csv" ] || echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q" > results/data.csv

cd build/
for alg in {1,3}
do
	for n in {16..26..2}
	do
		for q in {10..25..3}
		do
			for lr in {5..25..5}
			do
				if [ $lr -lt $n ]
				then
					./rtxrmq $RANDOM $dev $((2**$n)) $bs $((2**$q)) $((2**$lr)) $nt $alg
				fi
			done
			./rtxrmq $RANDOM $dev $((2**$n)) $((2**$q)) -1 $nt $alg
		done
	done
done

