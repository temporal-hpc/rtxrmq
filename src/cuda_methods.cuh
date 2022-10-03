#pragma once
// 1) CUDA Warp Shuffle
__inline__ __device__ int warp_min(int val){
    for (int offset = (WARPSIZE>>1); offset > 0; offset >>= 1) {
        int n = __shfl_down_sync(0xffffffff, val, offset, WARPSIZE);
        val = min(val, n);
    }
    return val;
}


// GPU parallel RMQs basic
__global__ void kernel_rmq_basic(int n, int q, float *x, int2 *rmq, float *out){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    int l = rmq[tid].x;
    int r = rmq[tid].y;
    float min = x[l];
    float val;
    for(int i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
        }
    }
    //printf("thread %i accessing out[%i] putting min %f\n", tid, tid, min);
    out[tid] = min;
}

__global__ void min_kernel(int n, float *x, float *out) {
    __shared__ int min_w[WARPSIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n){
        return;
    }
    if(threadIdx.x < WARPSIZE){
        min_w[threadIdx.x] = *(int*) &x[tid];
    }
    __syncthreads();
    //printf("#BASE thread %i  --->  min_w %f\n", tid, *(float*)&min_w[threadIdx.x]);
    int e = *(int*) &x[tid];
    int w = warp_min(e);
    if (threadIdx.x % WARPSIZE == 0){
	    min_w[threadIdx.x / WARPSIZE] = w;
        //printf("#WARP LEVEL thread %i  --->  warp, val %f,    min %f\n", tid, x[tid], *(float*)&w);
    }
    __syncthreads();
    if (threadIdx.x < WARPSIZE){
        w = warp_min(min_w[threadIdx.x]);
        //printf("#BLOCK LEVEL thread %i  --->  warp, val %f,    min %f\n", tid, *(float*)&min_w[threadIdx.x], *(float*)&w);
    }
    if(threadIdx.x == 0) {
        int *addr = *(int**) &out;
        //printf("BLOCK %i min %f\n", blockIdx.x, *(float*)&w);
        atomicMin(addr, w);
    }
}

void cudaWarpShuffle(int n, int k, int steps, float *darray, curandState *devStates) {
    printf("Simulating for %i steps\n", steps);
    float min, *dout, max=100000.0f;
    int grid = (n + BSIZE - 1) / BSIZE;
    cudaMalloc(&dout, sizeof(float)*1);

    for(int ki = 0; ki<steps; ++ki){
        // Warp Shuffle Min
        //print_array_dev(n, darray);
        //cpuprint_array(n, darray);
        cudaMemcpy(dout, &max, sizeof(float)*1, cudaMemcpyHostToDevice);
        printf(AC_BOLDCYAN "\tCUDA Warp Shuffle MIN................" AC_RESET);
        Timer timer;
        min_kernel<<<grid, BSIZE>>>(n, darray, dout);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        cudaMemcpy(&min, dout, sizeof(float), cudaMemcpyDeviceToHost);
        printf(AC_BOLDCYAN "done: %f ms (min %f, cpumin %f)\n" AC_RESET, timer.get_elapsed_ms(), min, cpumin_point(n, darray));

        // Simulation --> update points with CUDA kernel 
        printf("\tParticles Random Movement............"); fflush(stdout);
        dim3 block(BSIZE, 1, 1);
        dim3 grid((n + BSIZE-1)/BSIZE, 1, 1);
        timer.restart();
        kernel_point_simulation<<<grid, block>>>(n, darray, devStates);
        cudaDeviceSynchronize();
        timer.stop();
        printf("done: %f ms\n", timer.get_elapsed_ms());
        printf("\n");
    }
    printf("done\n");
}






// 2) CUB approach
void cudaCUB(int n, int k, int steps, float *darray, curandState *devStates){
    float *d_out, h_out;
    cub::CachingDeviceAllocator  g_allocator(true);
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1));
    // FIRST RUN IS TO KNOW temp_storage_bytes
    void    *d_temp_storage = NULL;
    size_t  temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, darray, d_out, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    printf("Simulating for %i steps\n", steps);
    for(int ki = 0; ki<steps; ++ki){
        // Run parallel MIN
        printf(AC_BOLDCYAN "\tCUB DeviceReduce::Min................" AC_RESET);
        Timer timer;
        CubDebugExit(cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, darray, d_out, n));
        cudaDeviceSynchronize();
        timer.stop();
        CubDebugExit(cudaMemcpy(&h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost));
        printf(AC_BOLDCYAN "done: %f ms (min %f, cpumin %f)\n" AC_RESET, timer.get_elapsed_ms(), h_out, cpumin_point(n, darray));

        // Simulation --> update points with CUDA kernel 
        printf("\tParticles Random Movement............"); fflush(stdout);
        dim3 block(BSIZE, 1, 1);
        dim3 grid((n + BSIZE-1)/BSIZE, 1, 1);
        timer.restart();
        kernel_point_simulation<<<grid, block>>>(n, darray, devStates);
        cudaDeviceSynchronize();
        timer.stop();
        printf("done: %f ms\n", timer.get_elapsed_ms());
        printf("\n");
    }
    printf("done\n");
}


// 3) Thrust approach
void cudaThrust(int n, int k, int steps, float *darray, curandState *devStates) {
    thrust::device_ptr<float> D(darray);
    printf("Simulating for %i steps\n", steps);
    for(int ki = 0; ki<steps; ++ki){
        // Thrust Min
        printf(AC_BOLDCYAN "\tThrust Min..........................." AC_RESET);
        Timer timer;
        float min = thrust::reduce(D, D+n, 10000.0, thrust::minimum<float>());
        cudaDeviceSynchronize();
        timer.stop();
        printf(AC_BOLDCYAN "done: %f ms (min %f, cpumin %f)\n" AC_RESET, timer.get_elapsed_ms(), min, cpumin_point(n, darray));

        // Simulation --> update points with CUDA kernel 
        printf("\tParticles Random Movement............"); fflush(stdout);
        dim3 block(BSIZE, 1, 1);
        dim3 grid((n + BSIZE-1)/BSIZE, 1, 1);
        timer.restart();
        kernel_point_simulation<<<grid, block>>>(n, darray, devStates);
        cudaDeviceSynchronize();
        timer.stop();
        printf("done: %f ms\n", timer.get_elapsed_ms());
        printf("\n");
    }
    printf("done\n");
}

// 4) GPU RMQ basic approach
float* gpu_rmq_basic(int n, int q, float *devx, int2 *devrmq){
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-11s).............." AC_RESET, algStr[ALG_GPU_BASE]); fflush(stdout);
    timer.restart();
    kernel_rmq_basic<<<grid, block>>>(n, q, devx, devrmq, dout);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float timems = timer.get_elapsed_ms();
    printf(AC_BOLDCYAN "done: %f secs: [%.2f RMQs/sec, %.3f usec/RMQ]\n" AC_RESET, timems/1000.0, (double)q/(timems/1000.0), (double)timems*1000.0/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    return hout;
}
