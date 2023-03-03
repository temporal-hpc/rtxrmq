#pragma once
#include <curand_kernel.h>
#include <random>
#include <cmath>

__global__ void kernel_setup_prng(int n, int seed, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id <= n){
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void kernel_random_array(int n, curandState *state, float *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    float x = curand_uniform(&state[id]);
    //array[id] = x*1000.0f;
    array[id] = x;
}

std::pair<float*, curandState*> create_random_array_dev(int n, int seed){
    // cuRAND states
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    // data array
    float* darray;
    cudaMalloc(&darray, sizeof(float)*n);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();

    // gen random numbers
    kernel_random_array<<<grid,block>>>(n, devStates, darray);
    cudaDeviceSynchronize();

    return std::pair<float*, curandState*>(darray,devStates);
}

__global__ void kernel_random_array(int n, int max, int lr, curandState *state, int2 *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    int y = lr > 0 ? lr : curand_uniform(&state[id]) * (max-1);
    int x = curand_uniform_double(&state[id]) * (max-y-1);
    array[id].x = x;
    array[id].y = x+y;
}

std::pair<int2*, curandState*> create_random_array_dev2(int n, int max, int lr, int seed){
    // cuRAND states
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    // data array
    int2* darray;
    cudaMalloc(&darray, sizeof(int2)*n);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();

    // gen random numbers
    kernel_random_array<<<grid,block>>>(n, max, lr, devStates, darray);
    cudaDeviceSynchronize();

    return std::pair<int2*, curandState*>(darray,devStates);
}

int gen_lr(int n, int lr, std::mt19937 gen) {
    if (lr > 0) {
        return lr;
    } else if (lr == -1) {
        std::uniform_int_distribution<int> dist(0, n-1);
        return dist(gen);
    } else if (lr == -2) {
        std::lognormal_distribution<double> dist(1.5, 1);
        int x = (int)(dist(gen) * pow(n, 0.7));
        //printf("after x  %i\n", x); fflush(stdout);
        while (x < 0 || x > n-1) {
            x = (int)(dist(gen) * pow(n, 0.7));
            //printf("in loop x  %i\n", x); fflush(stdout);
        }
        return x;
    } else if (lr == -3) {
        std::lognormal_distribution<double> dist(1.5, 1);
        int x = (int)(dist(gen) * pow(n, 0.3));
        while (x < 0 || x > n-1)
            x = (int)(dist(gen) * pow(n, 0.3));
        return x;
    }
    return 0;
}

int2* random_queries(int q, int lr, int n, int seed) {
    int2 *query = new int2[q];
    std::mt19937 gen(seed);

    for (int i = 0; i < q; ++i) {
        int qsize = gen_lr(n, lr, gen);
        //printf("qsize  %i\n", qsize); fflush(stdout);
        std::uniform_int_distribution<int> lrand(0, n-1 - (qsize-1));
        int l = lrand(gen);
        query[i].x = l;
        query[i].y = l + (qsize - 1);
        //printf("(l,r) -> (%i, %i)\n\n", query[i].x, query[i].y);
    }
    return query;
}
