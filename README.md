# RTXRMQ - Batch Range Minimum Queries Accelerated by RTX
RMQ implementation using Nvidia OptiX and RT cores. This repository is part of our work `Accelerating range minimum queries with ray tracing cores` published on Elsevier's FGCS Journal. 

DOI: https://doi.org/10.1016/j.future.2024.03.040

The results in the paper correspond to algorithm 5 (RTX_blocks).

## Dependencies
- CUDA 11 or later
- OptiX 7.7 or later

## Compile and run
```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=<PATH-TO-OPTIX-MAIN-DIR>
make
./rtxrmq <n> <q> <lr> <alg>

n   = num elements
q   = num RMQ querys
lr  = length of range; min 1, max n
  >0 -> value
  -1 -> uniform distribution (large values)
  -2 -> lognormal distribution (medium values)
  -3 -> lognormal distribution (small values)
alg = algorithm
   0 -> [CPU] BASE
   1 -> [CPU] HRMQ
   2 -> [GPU] BASE
   3 -> [GPU] RTX_cast
   4 -> [GPU] RTX_trans
   5 -> [GPU] RTX_blocks (RTXRMQ)
   6 -> [GPU] RTX_lup
   7 -> [GPU] LCA
   8 -> [GPU] RTX_ias
   9 -> [GPU] RTX_ias_trans
   100, 101, 102, 103, 105 -> algs 1 2 3 5 returning indices

Options:
   --bs <block size>         block size for RTX_blocks (default: 2^15)
   --nb <#blocks>            number of blocks for RTX_blocks (overrides --bs)
   --reps <repetitions>      RMQ repeats for the avg time (default: 10)
   --dev <device ID>         device ID (default: 0)
   --nt  <thread num>        number of CPU threads
   --seed <seed>             seed for PRNG
   --check                   check correctness
   --save-time=<file>
   --save-power=<file>
```


## Example compilation and execution
```
cmake .. -DOPTIX_HOME=~/opt/optix/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
make
rtxrun ./rtxrmq $((10**7)) $((10**7)) -3 5
```


## References
[1] E. Meneses, C. Navarro, H. Ferrada, F. Quezada, Accelerating range minimum queries with ray tracing cores, Future Generation Computer Systems 157 (2024) 98-111

[2] H. Ferrada, G. Navarro, Improved range minimum queries, J. Discrete Algorithms 43 (2017) 72–80

[3] A. Polak, A. Siwiec, M. Stobierski, Euler meets GPU: Practical graph algorithms with theoretical guarantees, in: 2021 IEEE International Parallel and Distributed Processing Symposium, IPDPS, IEEE, 2021, pp. 233–244
