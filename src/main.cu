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
#define RTX_REPEATS 10
#define ALG_CPU_BASE        0
#define ALG_CPU_HRMQ        1
#define ALG_GPU_BASE        2
#define ALG_GPU_RTX_CAST    3
#define ALG_GPU_RTX_TRANS   4
#define ALG_GPU_RTX_BLOCKS  5
#define ALG_GPU_RTX_LUP     6

// TODO add other state-of-the-art GPU rmq algs
const char *algStr[7] = { "[CPU] BASE", "[CPU] HRMQ", "[GPU] BASE", "[GPU] RTX_cast", "[GPU] RTX_trans", "[GPU] RTX_blocks", "[GPU] RTX_lup"}; 


#define SAVE 0
#define SAVE_FILE "../results/data.csv"
#ifdef CHECK
     #define CHECK 1
#else
     #define CHECK 1
#endif
#define MEASURE_POWER 0
#if MEASURE_POWER == 1
    #define REPS 100
#else
    #define REPS 10
#endif


#include "common/common.h"
#include "common/Timer.h"
#include "common/nvmlPower.hpp"
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
    int bs = atoi(argv[4]);
    int q = atoi(argv[5]);
    int lr = atoi(argv[6]);
    int nt = atoi(argv[7]);
    int alg = atoi(argv[8]);
    if (lr >= n) {
        fprintf(stderr, "Error: lr can not be bigger than n\n");
        return -1;
    }

    printf( "Params:\n"
            "   seed = %i\n"
            "   dev = %i\n"
            AC_GREEN "   n   = %i (~%f GB, float)\n" AC_RESET
            "   bs = %i\n"
            AC_GREEN "   q   = %i (~%f GB, int2)\n" AC_RESET
            "   lr  = %i\n"
            "   nt  = %i CPU threads\n"
            "   alg = %i (%s)\n\n",
            seed, dev, n, sizeof(float)*n/1e9, bs, q, sizeof(int2)*q/1e9, lr, nt, alg, algStr[alg]);
    cudaSetDevice(dev);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    Timer timer;
    printf(AC_YELLOW "Generating n=%i values..............", n); fflush(stdout);
    std::pair<float*, curandState*> p = create_random_array_dev(n, seed);
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    timer.restart();
    printf(AC_YELLOW "Generating q=%i queries.............", q); fflush(stdout);
    std::pair<int2*, curandState*> qs = create_random_array_dev2(q, n, lr, seed+7); //TODO use previous states
    printf("done: %f secs\n" AC_RESET, timer.get_elapsed_ms()/1000.0f);


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

    
    write_results(dev, alg, n, bs, q, lr);
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
        default: // RTX algs
            out = rtx_rmq(alg, n, bs, q, p.first, qs.first, p.second, dev);
            break;
    }

    if (CHECK){
        printf("\nCHECKING RESULT:\n");
        float *expected = gpu_rmq_basic(n, q, p.first, qs.first);
        //float *expected = cpu_rmq<float>(n, q, hA, hQ, nt);
        //hAi = reinterpret_cast<int*>(hA);
        //outi = rmq_rmm_par(n, q, hAi, hQ, nt);
        //float *expected = reinterpret_cast<float*>(outi);
        printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
        int pass = check_result(hA, hQ, q, expected, out);
        printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");
        //for (int i = 0; i < 101; ++i) {
            //printf("%f ", hA[i+33554332]);
            //if (i%10==9) printf("\n");
        //}
        //printf("\n");
    }

    printf("Benchmark Finished\n");
    return 0;
}
