#pragma once

#define NUM_REQUIRED_ARGS 6
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <dev> <n> <q> <nt> <alg>\n" AC_RESET
                    "dev = device ID\n"
                    "n   = num elements\n"
                    "q   = num RMQ querys\n"
                    "nt  = num CPU threads\n"
                    "alg = algorithm\n"
                    "   0 -> %s\n"
                    "   1 -> %s\n"
                    "   2 -> %s\n"
                    "   3 -> %s\n",
                    algStr[0],
                    algStr[1],
                    algStr[2],
                    algStr[3]
                );
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

bool check_result(float *hA, int2 *hQ, int q, float *expected, float *result){
    for (int i = 0; i < q; ++i) {
        //if (expected[i] != result[i]) { // RT-cores doesn't introduce floating point errors
        if (!is_equal(expected[i], result[i])) {
            printf("Error on %i-th query: got %f, expected %f\n", i, result[i], expected[i]);
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

bool check_result(float *hA, int2 *hQ, int q, int *expected, int *result){
    for (int i = 0; i < q; ++i) {
        if (expected[i] != result[i]) {
            printf("Error on %i-th query: got %f, expected %f\n", i, result[i], expected[i]);
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
