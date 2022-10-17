dev=0
nt=8

cd build/
for alg in {1,3}
do
	for n in {16..26..2}
	do
		for q in {10..25..3}
		do
			for lr in {5,10,20}
			do
				if [ $lr -lt $n ]
				then
					./rtxrmq $RANDOM $dev $((2**$n)) $((2**$q)) $((2**$lr)) $nt $alg
				fi
			done
			./rtxrmq $RANDOM $dev $((2**$n)) $((2**$q)) -1 $nt $alg
		done
	done
done

