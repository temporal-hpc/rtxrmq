#pragma once
#include <unistd.h>

#define NUM_REQUIRED_ARGS 9
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <seed> <dev> <n> <q> <lr> <nt> <alg>\n" AC_RESET
                    "seed = seed for PRNG\n"
                    "dev = device ID\n"
                    "n   = num elements\n"
                    "bs  = block size for RTX_blocks\n"
                    "q   = num RMQ querys\n"
                    "lr  = size of range (-1: randomized)\n"
                    "nt  = num CPU threads\n"
                    "alg = algorithm\n"
                    "   0 -> %s\n"
                    "   1 -> %s\n"
                    "   2 -> %s\n"
                    "   3 -> %s\n"
                    "   4 -> %s\n"
                    "   5 -> %s\n",
                    algStr[0],
                    algStr[1],
                    algStr[2],
                    algStr[3],
                    algStr[4],
                    algStr[5]
                );
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

bool check_result(float *hA, int2 *hQ, int q, float *expected, float *result){
    bool pass = true;
    for (int i = 0; i < q; ++i) {
        //if (expected[i] != result[i]) { // RT-cores don't introduce floating point errors
        if (!is_equal(expected[i], result[i])) {
            printf("Error on %i-th query: got %f, expected %f\n", i, result[i], expected[i]);
            printf("  [%i,%i]\n", hQ[i].x, hQ[i].y);
            pass = false;
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    //for (int j = 0; j <= 1<<24; ++j) {
    //    printf("%f\n", hA[j]);
    //}
    return pass;
}

bool check_result(float *hA, int2 *hQ, int q, int *expected, int *result){
    for (int i = 0; i < q; ++i) {
        if (expected[i] != result[i]) {
            printf("Error on %i-th query: got %i, expected %i\n", i, result[i], expected[i]);
            //printf("[%i,%i]\n", hQ[i].x, hQ[i].y);
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    return true;
}

bool check_parameters(int argc){
    if(argc < NUM_REQUIRED_ARGS){
        fprintf(stderr, AC_YELLOW "missing arguments\n" AC_RESET);
        print_help();
        return false;
    }
    return true;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);
    printf("  Memory Clock Rate (MHz):      %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits):      %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void write_results(int dev, int alg, int n, int q, int lr) {
    if (!SAVE) return;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    char *device = prop.name;
    if (alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
        strcpy(device, "CPU ");
        char hostname[50];
        gethostname(hostname, 50);
        strcat(device, hostname);
    }

    FILE *fp;
    fp = fopen(SAVE_FILE, "a");
    fprintf(fp, "%s,%s,%i,%i,%i,%i",
            device,
            algStr[alg],
            REPS,
            n,
            q,
            lr);
    fclose(fp);
}

void write_results(float time_ms, int q) {
    if (!SAVE) return;
    float time_it = time_ms/REPS;
    FILE *fp;
    fp = fopen(SAVE_FILE, "a");
    fprintf(fp, ",%f,%f,%f\n",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q);
    fclose(fp);
}
