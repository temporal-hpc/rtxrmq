#pragma once
// Kernel RMQs basic
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

__global__ void kernel_rmq_basic(int n, int q, float *x, int2 *rmq, float *out, int *indices){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    int l = rmq[tid].x;
    int r = rmq[tid].y;
    float min = x[l];
    float val;
    int idx_min = l;
    for(int i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
            idx_min = i;
        }
    }
    //printf("thread %i accessing out[%i] putting min %f\n", tid, tid, min);
    out[tid] = min;
    indices[tid] = idx_min;
}

// GPU RMQ basic approach
float* gpu_rmq_basic(int n, int q, float *devx, int2 *devrmq, CmdArgs args){
    int reps = args.reps;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    size_t bytesUsed = sizeof(int)*n;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[ALG_GPU_BASE], reps); fflush(stdout);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        kernel_rmq_basic<<<grid, block>>>(n, q, devx, devrmq, dout);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0, reps, args, {bytesUsed, 0});
    return hout;
}

float* gpu_rmq_basic(int n, int q, float *devx, int2 *devrmq, CmdArgs args, int* &indices){
    int reps = args.reps;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    indices = (int*)malloc(sizeof(int)*q);
    int *d_indices;
    cudaMalloc(&d_indices, sizeof(int)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[ALG_GPU_BASE], reps); fflush(stdout);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        kernel_rmq_basic<<<grid, block>>>(n, q, devx, devrmq, dout, d_indices);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices, d_indices, sizeof(int)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(d_indices));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0, reps, args);
    return hout;
}
