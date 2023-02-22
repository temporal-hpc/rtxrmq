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

    CmdArgs args = get_args(argc, argv);
    int reps = args.reps;
    int seed = args.seed;
    int dev = args.dev;
    int n = args.n;
    int bs = args.bs;
    int q = args.q;
    int lr = args.lr;
    int nt = args.nt;
    int alg = args.alg;

    cudaSetDevice(dev);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    float *hA, *dA;
    int *hAi;
    int2 *hQ, *dQ;


    Timer timer;
    printf(AC_YELLOW "Generating n=%-10i values............", n); fflush(stdout);
    std::pair<float*, curandState*> p = create_random_array_dev(n, seed);
    dA = p.first;
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    timer.restart();
    printf(AC_YELLOW "Generating q=%-10i queries...........", q); fflush(stdout);
    //std::pair<int2*, curandState*> qs = create_random_array_dev2(q, n, lr, seed+7); //TODO use previous states
    //dQ = qs.first;
    hQ = random_queries(q, lr, n, seed);
    cudaMalloc(&dQ, sizeof(int2)*q);
    cudaMemcpy(dQ, hQ, sizeof(int2)*q, cudaMemcpyHostToDevice);
    printf("done: %f secs\n" AC_RESET, timer.get_elapsed_ms()/1000.0f);


    // 1.5 data on CPU
    if (args.check || alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
        hA = new float[n];
        //hQ = new int2[q];
        cudaMemcpy(hA, p.first, sizeof(float)*n, cudaMemcpyDeviceToHost);
        //cudaMemcpy(hQ, qs.first, sizeof(int2)*q, cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    write_results(dev, alg, n, bs, q, lr, reps, args);
    // 2) computation
    float *out;
    int *outi;

    switch(alg){
        case ALG_CPU_BASE:
            out = cpu_rmq<float>(n, q, hA, hQ, nt, args);
            break;
        case ALG_CPU_HRMQ:
            hAi = reinterpret_cast<int*>(hA);
            outi = rmq_rmm_par(n, q, hAi, hQ, nt, args);
            out = reinterpret_cast<float*>(outi);
            break;
        case ALG_GPU_BASE:
            out = gpu_rmq_basic(n, q, dA, dQ, args);
            break;
        default:
            out = rtx_rmq(alg, n, bs, q, dA, dQ, args);
            break;
    }


    if (args.check){
        printf("\nCHECKING RESULT:\n");
        args.reps = 1;
        args.save_time = 0;
        args.save_power = 0;
        float *expected = gpu_rmq_basic(n, q, dA, dQ, args);
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
