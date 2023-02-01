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

// GPU RMQ basic approach
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
    printf(AC_BOLDCYAN "done: %f secs: [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0);
    return hout;
}
