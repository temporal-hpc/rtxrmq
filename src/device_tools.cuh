#pragma once
#define PRINT_LIMIT 32


__global__ void kernel_print_array_dev(int n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %f\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_vertices_dev(int ntris, float3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+0, v[3*i+0].x, v[3*i+0].y, v[3*i+0].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+1, v[3*i+1].x, v[3*i+1].y, v[3*i+1].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+2, v[3*i+2].x, v[3*i+2].y, v[3*i+2].z);
        printf("\n");
    }
    if(i < ntris){
        printf("...\n");
    }
}

void print_array_dev(int n, float *darray){
    printf("Printing random array:\n");
    kernel_print_array_dev<<<1,1>>>(n, darray);
    cudaDeviceSynchronize();
}

void print_vertices_dev(int ntris, float3 *devVertices){
    printf("Printing vertices:\n");
    kernel_print_vertices_dev<<<1,1>>>(ntris, devVertices);
    cudaDeviceSynchronize();
}

__device__ float transformLR(int alg, int x, int N) {
    switch (alg) {
        case ALG_GPU_RTX_CAST:
            return (float)x / N; 
        case ALG_GPU_RTX_TRANS:
            int E = x / (1<<23);
            int M = x % (1<<23);
            float m = (float)(M + (1<<23)) / (1<<24);
            float f = m * (1 << E);
            return f;
    }
}

__global__ void kernel_gen_vertices(int alg, int N, float *array, float3 *vertices){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N){
        int k = 3*idx;
        float val = array[idx];
        // ray hits min on coord (val, l, r)
        float l = transformLR(alg, idx+1, N);
        float r = transformLR(alg, idx-1, N);
        float n = transformLR(alg, N, N);

        vertices[k+0] = make_float3(val, l, r);
        vertices[k+1] = make_float3(val, l, 2*n);
        vertices[k+2] = make_float3(val, -1*n, r);
    }
}

__global__ void kernel_gen_vertices_blocks(int num_blocks, int N, float *min_blocks, float *array, float3 *vertices){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = 3*idx;
    if(idx < num_blocks){
        float val = min_blocks[idx];
        // ray hits min on coord (val, l, r)
        float l = (float)(idx+1)/num_blocks;
        float r = (float)(idx-1)/num_blocks;
        float n = 1;

        vertices[k+0] = make_float3(val, l, r);
        vertices[k+1] = make_float3(val, l, 2*n);
        vertices[k+2] = make_float3(val, -1*n, r);
    } else if (idx < N) {
        int sub_idx = idx - num_blocks;
        int bid = sub_idx / RTX_BLOCK_SIZE;
        int lid = sub_idx % RTX_BLOCK_SIZE;
        float val = array[sub_idx];

        float l = (float)(lid+1)/RTX_BLOCK_SIZE + 2*(bid+1);
        float r = (float)(lid-1)/RTX_BLOCK_SIZE;

        vertices[k+0] = make_float3(val, l, r);
        vertices[k+1] = make_float3(val, l, 2);
        vertices[k+2] = make_float3(val, 2*bid+1, r);
        //printf("%i-th element %f  at  %f,  %f\n", sub_idx, val, l, r);
    }
}

__global__ void kernel_transform_querys(int alg, float2 *Qf, int2 *Qi, int q, int N) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= q) return;
    float l = transformLR(alg, Qi[tid].x, N);
    float r = transformLR(alg, Qi[tid].y, N);
    Qf[tid] = make_float2(l,r);
}

float2 *transform_querys(int alg, int2 *Q, int q, int N) {
    float2 *querys;
    cudaMalloc(&querys, sizeof(float2)*q);
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    kernel_transform_querys<<<grid, block>>>(alg, querys, Q, q, N);
    cudaDeviceSynchronize();
    return querys;
}

__global__ void kernel_gen_triangles(int ntris, float *array, uint3 *triangles){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        triangles[tid] = make_uint3(k, k+1, k+2);
    }
}

float3* gen_vertices_dev(int alg, int ntris, float *darray){
    // vertices data
    float3 *devVertices;
    cudaMalloc(&devVertices, sizeof(float3)*3*ntris);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices<<<grid, block>>>(alg, ntris, darray, devVertices);
    cudaDeviceSynchronize();
    return devVertices;
}

__global__ void kernel_min_blocks(float *min_blocks, float *darray, int num_blocks, int N) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    int first = tid * RTX_BLOCK_SIZE;
    int min = darray[first];
    for (int i = 1; i < RTX_BLOCK_SIZE && i < N - first; ++i) {
        if (darray[i+first] < min)
            min = darray[i+first];
    }
    min_blocks[tid] = min;
}

float3* gen_vertices_blocks_dev(int N, float *darray){
    // create array with mins of each block
    int num_blocks = (N+RTX_BLOCK_SIZE-1) / RTX_BLOCK_SIZE;
    int ntris = N + num_blocks;

    float *min_blocks;
    cudaMalloc(&min_blocks, sizeof(float)*num_blocks);
    dim3 block(BSIZE, 1, 1);
    dim3 grid_mins((num_blocks+BSIZE-1)/BSIZE,1,1);
    kernel_min_blocks<<<grid_mins, block>>>(min_blocks, darray, num_blocks, N);
    CUDA_CHECK( cudaDeviceSynchronize() );

    // vertices data
    float3 *devVertices;
    cudaMalloc(&devVertices, sizeof(float3)*3*ntris);

    // setup states
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices_blocks<<<grid, block>>>(num_blocks, ntris, min_blocks, darray, devVertices);
    CUDA_CHECK( cudaDeviceSynchronize() );
    return devVertices;
}

uint3* gen_triangles_dev(int ntris, float *darray){
    // data array
    uint3 *devTriangles;
    cudaMalloc(&devTriangles, sizeof(uint3)*ntris);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_triangles<<<grid, block>>>(ntris, darray, devTriangles);
    cudaDeviceSynchronize();
    return devTriangles;
}

float cpumin_vertex(int nv, float3 *dv){
    float3 *hv = new float3[nv];
    cudaMemcpy(hv, dv, sizeof(float3)*nv, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float min = 1000.0f;
    for(int i=0; i<nv; ++i){
        float val = hv[i].x;
        if(val < min){
            min = val;
        }
    }
    return min;
}

float cpurmq_vertex(int nv, float3 *dv, int nq, int2 *dq, int qi){
    float3 *hv = new float3[nv];
    int2 *hq = new int2[nq];
    cudaMemcpy(hv, dv, sizeof(float3)*nv, cudaMemcpyDeviceToHost);
    cudaMemcpy(hq, dq, sizeof(int2)*nq, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float min = 10000.0f;
    int l = hq[qi].x;
    int r = hq[qi].y;
    for(int i=3*l; i<=3*r; ++i){
        float val = hv[i].x;
        if(val < min){
            min = val;
        }
    }
    return min;
}


float cpumin_point(int np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float min = 1000000.0f;
    for(int i=0; i<np; ++i){
        //printf("Checking [%i] = %f\n", i, hp[i]);
        float val = hp[i];
        if(val < min){
            min = val;
            //printf("CPU new min %f\n", min);
        }
    }
    return min;
}

void cpuprint_array(int np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<np; ++i){
        printf("array [%i] = %f\n", i, hp[i]);
    }
}
