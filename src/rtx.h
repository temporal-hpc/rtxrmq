#pragma once

int* compute_min_blocks(int n, float* d_array, int num_blocks, int block_size) {
    int2 *queries, *d_queries;
    queries = (int2*)malloc(num_blocks * sizeof(int2));
    for (int i = 0; i < num_blocks; ++i) {
        queries[i] = make_int2(i*block_size, (i+1)*block_size-1);
    }
    CUDA_CHECK( cudaMalloc(&d_queries, num_blocks * sizeof(int2)) );
    CUDA_CHECK( cudaMemcpy(d_queries, queries, num_blocks * sizeof(int2), cudaMemcpyHostToDevice) );

    int* min_blocks;
    CUDA_CHECK( cudaMalloc(&min_blocks, num_blocks * sizeof(int)) );

    dim3 block(BSIZE, 1, 1);
    dim3 grid((num_blocks+BSIZE-1)/BSIZE, 1, 1);

    kernel_rmq_basic_idx<<<grid, block>>>(n, num_blocks, d_array, d_queries, min_blocks);
    CUDA_CHECK( cudaDeviceSynchronize() );

    return min_blocks;
}

template <typename T>
T* rtx_rmq(int alg, int n, int bs, int q, float *darray, int2 *dquery, CmdArgs args) {
    int dev = args.dev;
    int reps = args.reps;
    Timer timer;
    T *output, *d_output;
    //float cpuMin=-1.0f;
    output = (T*)malloc(q*sizeof(T));

    // 1) Generate geometry from device data
    printf("Generating geometry......................."); fflush(stdout);
    timer.restart();
    float3 *devVertices;
    //int orig_n;
    float *LUP = nullptr;
    int num_blocks;
    if (alg == ALG_GPU_RTX_BLOCKS || alg == ALG_GPU_RTX_BLOCKS_IDX) {
        devVertices = gen_vertices_blocks_dev(n, bs, darray);
        num_blocks = (n+bs-1) / bs;
        //orig_n = n;
        n += num_blocks;
    } else if (alg == ALG_GPU_RTX_IAS) {
        devVertices = gen_vertices_blocks_dev(n, bs, darray);
        num_blocks = (n+bs-1) / bs;
        //orig_n = n;
        n += num_blocks;
    } else if (alg == ALG_GPU_RTX_IAS_TRANS) {
        devVertices = gen_vertices_blocks_dev_ias(n, bs, darray);
        num_blocks = (n+bs-1) / bs;
        //orig_n = n;
        n += num_blocks;
    } else if (alg == ALG_GPU_RTX_LUP) {
        devVertices = gen_vertices_lup_dev(n, bs, darray, LUP);
    } else {
        devVertices = gen_vertices_dev(alg, n, darray);
    }
    uint3 *devTriangles = gen_triangles_dev(n, darray);
    //print_array_dev(n, darray);
    //print_vertices_dev(n, devVertices);
    timer.stop();
    float geom_time = timer.get_elapsed_ms();
    printf("done: %f ms\n",geom_time); fflush(stdout);

    int *min_blocks;
    if (alg == ALG_GPU_RTX_BLOCKS_IDX)
        min_blocks = compute_min_blocks(n, darray, num_blocks, bs);

    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state, args);
    //if (alg == ALG_GPU_RTX_BLOCKS || alg == ALG_GPU_RTX_IAS || alg == ALG_GPU_RTX_IAS_TRANS)
    //    createGroupsClosestHit_Blocks(state);
    //else if (alg == ALG_GPU_RTX_LUP)
    //    createGroupsClosestHit_LUP(state);
    //else
    //    createGroupsClosestHit(state);
    createProgramGroups(state, alg);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // 3) Build Acceleration Structure 
    printf("%sBuild AS on GPU...........................", AC_MAGENTA); fflush(stdout);
    VBHMem mem = {0, 0};
    timer.restart();
    if (alg == ALG_GPU_RTX_IAS || alg == ALG_GPU_RTX_IAS_TRANS)
        buildIAS(mem, state, 3*n, n, devVertices, devTriangles, bs, num_blocks,alg);
    else
        buildASFromDeviceData(mem, state, 3*n, n, devVertices, devTriangles);
    cudaDeviceSynchronize();
    timer.stop();
    float AS_time = timer.get_elapsed_ms();
    printf("done: %f ms [output: %f MB, temp %f MB]\n" AC_RESET, AS_time, mem.out_buffer/1e6, mem.temp_buffer/1e6);

    // 4) Populate and move parameters to device (ONCE)
    CUDA_CHECK( cudaMalloc(&d_output, q*sizeof(T)) );
    timer.restart();
    Params params;
    Params *device_params;

    params.handle = state.gas_handle;
    params.min = -1.0f;
    params.max = 10.0f;
    params.output = alg < 100 ? (float*)d_output : nullptr;
    params.idx_output = alg < 100 ? nullptr : (int*)d_output;

    if (alg == ALG_GPU_RTX_BLOCKS || alg == ALG_GPU_RTX_IAS || alg == ALG_GPU_RTX_IAS_TRANS || alg == ALG_GPU_RTX_BLOCKS_IDX) {
        params.query = nullptr;
        params.iquery = dquery;
        params.num_blocks = ceil(sqrt(num_blocks + 1));
        params.block_size = bs;
        params.nb = num_blocks;
        if (alg == ALG_GPU_RTX_BLOCKS_IDX)
            params.min_block = min_blocks;
    } else if (alg == ALG_GPU_RTX_LUP) {
        params.query = nullptr;
        params.iquery = dquery;
        num_blocks = (n + bs - 1) / bs;
        params.num_blocks = ceil(sqrt(num_blocks));
        params.nb = num_blocks;
        params.block_size = bs;
        params.LUP = LUP;
    } else {
        params.query = transform_querys(alg, dquery, q, n);
        params.iquery = nullptr;
    }
    printf("(%7.3f MB).........", (double)sizeof(Params)/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Computation
    if (alg < 100) {
        printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[alg], reps); fflush(stdout);
    } else {
        printf(AC_BOLDCYAN "Computing RMQs index (%-16s,r=%-3i)..." AC_RESET, algStr[alg%10], reps); fflush(stdout);
    }
    //printf("\n");
    if (args.save_power)
        GPUPowerBegin(algStr[alg], 100, dev, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, q, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    if (args.save_power)
        GPUPowerEnd();
    float timems = timer.get_elapsed_ms();
    CUDA_CHECK( cudaMemcpy(output, d_output, q*sizeof(T), cudaMemcpyDeviceToHost) );
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/avg_time, (double)avg_time*1e9/q);
    write_results(timems, q, geom_time + AS_time, reps, args, mem);
        
    // 6) clean up
    printf("cleaning up RTX environment..............."); fflush(stdout);
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    for (int i = 0; i < 3; ++i) {
        OPTIX_CHECK(optixProgramGroupDestroy(state.program_groups[i]));
    }
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(device_params));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    printf("done: %f ms\n", timer.get_elapsed_ms());
    return output;
}

