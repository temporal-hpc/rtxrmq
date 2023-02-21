dev=0
#n=$((2**26))
q=$((2**26))
#lr=-1
nt=8
alg=5
reps=10

[ ! -f "results/data.csv" ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ../results/3D_plot.csv

cd ../build/
for n in {16..26}
do
	for nb in {1..12}
	do
		for lr in {1..15}
		do
			./rtxrmq $((2**$n)) $q $((2**$n / 2**$lr)) $alg --bs $((2**$n / 2**$nb)) --reps $reps --nt $nt --dev $dev --save-time ../results/3D_plot.csv
		done
	done
done

