#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#define BSIZE 1024
#define WARPSIZE 32
#define RTX_REPEATS 1
#define ALG_CPU_BASE        0
#define ALG_CPU_HRMQ        1
#define ALG_GPU_BASE        2
#define ALG_GPU_RTX         3

// TODO add other state-of-the-art GPU rmq algs
const char *algStr[4] = { "[CPU] BASE", "[CPU] HRMQ", "[GPU] BASE", "[GPU] RTX"}; 


#define CHECK 1
#include "common/common.h"
#include "common/Timer.h"
#include "src/rand.cuh"
#include "src/tools.h"
#include "src/device_tools.cuh"
#include "src/device_simulation.cuh"
#include "src/cpu_methods.h"
#include "src/cuda_methods.cuh"
#include "src/rtx_functions.h"
#include "src/rtx.h"


int main(int argc, char *argv[]) {
    printf("----------------------------------\n");
    printf("        RTX-RMQ by Temporal       \n");
    printf("----------------------------------\n");
    if(!check_parameters(argc)){
        exit(EXIT_FAILURE);
    }
    int seed = atoi(argv[1]);
    int dev = atoi(argv[2]);
    int n = atoi(argv[3]);
    int q = atoi(argv[4]);
    int nt = atoi(argv[5]);
    int alg = atoi(argv[6]);
    printf( "Params:\n"
            "   seed = %i\n"
            "   dev = %i\n"
            AC_GREEN "   n   = %i (~%f GB, float)\n" AC_RESET
            AC_GREEN "   q   = %i (~%f GB, int2)\n" AC_RESET
            "   nt  = %i CPU threads\n"
            "   alg = %i (%s)\n\n",
            seed, dev, n, sizeof(float)*n/1e9, q, sizeof(int2)*n/1e9, nt, alg, algStr[alg]);
    cudaSetDevice(dev);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    std::pair<float*, curandState*> p = create_random_array_dev(n, seed);
    std::pair<int2*, curandState*> qs = create_random_array_dev2(q, n, seed+7); //TODO use previous states

    // 1.5 data on CPU
    float *hA;
    int *hAi;
    int2 *hQ;
    if (CHECK || alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
        hA = new float[n];
        hQ = new int2[q];
        cudaMemcpy(hA, p.first, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(hQ, qs.first, sizeof(int2)*q, cudaMemcpyDeviceToHost);
    }

    

    // 2) computation
    float *out;
    int *outi;

    switch(alg){
        case ALG_CPU_BASE:
            out = cpu_rmq<float>(n, q, hA, hQ, nt);
            break;
        case ALG_CPU_HRMQ:
            hAi = reinterpret_cast<int*>(hA);
            outi = rmq_rmm_par(n, q, hAi, hQ, nt);
            out = reinterpret_cast<float*>(outi);
            break;
        case ALG_GPU_BASE:
            out = gpu_rmq_basic(n, q, p.first, qs.first);
            break;
        case ALG_GPU_RTX:
            out = rtx_rmq(n, q, p.first, qs.first, p.second);
            break;
    }

    if (CHECK) {
        //float *expected = gpu_rmq_basic(n, q, p.first, qs.first);
        hAi = reinterpret_cast<int*>(hA);
        outi = rmq_rmm_par(n, q, hAi, hQ, nt);
        float *expected = reinterpret_cast<float*>(outi);
        check_result(hA, hQ, q, expected, out);
    }

    printf("Benchmark Finished\n");
    return 0;
}
