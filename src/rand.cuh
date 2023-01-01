#pragma once
#include <curand_kernel.h>

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
    //array[id] = x*100.0f+1.0f;
    array[id] = x + 0.0001;
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
    int y = lr > 0 ? lr : curand_uniform(&state[id]) * (max/100);
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

