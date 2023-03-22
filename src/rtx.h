#pragma once
float* rtx_rmq(int alg, int n, int bs, int q, float *darray, int2 *dquery, CmdArgs args) {
    int dev = args.dev;
    int reps = args.reps;
    Timer timer;
    float *output, *d_output;
    //float cpuMin=-1.0f;
    output = (float*)malloc(q*sizeof(float));

    // 1) Generate geometry from device data
    printf("Generating geometry......................."); fflush(stdout);
    timer.restart();
    float3 *devVertices;
    int orig_n;
    float *LUP = nullptr;
    int num_blocks;
    if (alg == ALG_GPU_RTX_BLOCKS || alg == ALG_GPU_RTX_IAS) {
        devVertices = gen_vertices_blocks_dev(n, bs, darray);
        num_blocks = (n+bs-1) / bs;
        orig_n = n;
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
    printf("done: %f ms\n",geom_time);

    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state);
    if (alg == ALG_GPU_RTX_BLOCKS)
        createGroupsClosestHit_Blocks(state);
    else if (alg == ALG_GPU_RTX_LUP)
        createGroupsClosestHit_LUP(state);
    else
        createGroupsClosestHit(state);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // 3) Build Acceleration Structure 
    printf("%sBuild AS on GPU...........................", AC_MAGENTA); fflush(stdout);
    timer.restart();
    if (alg == ALG_GPU_RTX_IAS)
        buildIAS(state, 3*n, n, devVertices, devTriangles, bs, num_blocks);
    else
        buildASFromDeviceData(state, 3*n, n, devVertices, devTriangles);
    cudaDeviceSynchronize();
    timer.stop();
    float AS_time = timer.get_elapsed_ms();
    printf("done: %f ms%s\n", AS_time, AC_RESET);

    // 4) Populate and move parameters to device (ONCE)
    printf("device params to GPU "); fflush(stdout);
    CUDA_CHECK( cudaMalloc(&d_output, q*sizeof(float)) );
    timer.restart();
    Params params;
    Params *device_params;

    params.handle = state.gas_handle;
    params.min = -1.0f;
    params.max = 2.0f;
    params.output = d_output;
    if (alg == ALG_GPU_RTX_BLOCKS) {
        params.query = nullptr;
        params.iquery = dquery;
        params.num_blocks = ceil(sqrt(num_blocks + 1));
        params.block_size = bs;
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

    // 5) Computation
    printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[alg], reps); fflush(stdout);
    //printf("\n");
    if (args.save_power)
        GPUPowerBegin(algStr[alg], 100, dev, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, q, 1, 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    if (args.save_power)
        GPUPowerEnd();
    float timems = timer.get_elapsed_ms();
    CUDA_CHECK( cudaMemcpy(output, d_output, q*sizeof(float), cudaMemcpyDeviceToHost) );
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/avg_time, (double)avg_time*1e9/q);
    write_results(timems, q, geom_time + AS_time, reps, args);
        
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
