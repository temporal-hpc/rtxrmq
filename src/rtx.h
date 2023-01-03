#pragma once
float* rtx_rmq(int alg, int n, int bs, int q, float *darray, int2 *dquery, curandState *devStates) {
    Timer timer;
    float *output, *d_output;
    //float cpuMin=-1.0f;
    output = (float*)malloc(q*sizeof(float));

    // 1) Generate geometry from device data
    printf("Generating geometry......................."); fflush(stdout);
    timer.restart();
    float3 *devVertices;
    int orig_n;
    if (alg != ALG_GPU_RTX_BLOCKS) {
        devVertices = gen_vertices_dev(alg, n, darray);
    } else {
        devVertices = gen_vertices_blocks_dev(n, bs, darray); // TODO implement
        int num_blocks = (n+bs-1) / bs;
        orig_n = n;
        n += num_blocks;
    }
    uint3 *devTriangles = gen_triangles_dev(n, darray);
    //print_array_dev(n, darray);
    //print_vertices_dev(n, devVertices);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state);
    if (alg != ALG_GPU_RTX_BLOCKS)
        createGroupsClosestHit(state);
    else
        createGroupsClosestHit_Blocks(state);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // 3) Build Acceleration Structure 
    printf("%sBuild AS on GPU...........................", AC_MAGENTA); fflush(stdout);
    timer.restart();
    buildASFromDeviceData(state, 3*n, n, devVertices, devTriangles);
    cudaDeviceSynchronize();
    timer.stop();
    printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);

    // 4) Populate and move parameters to device (ONCE)
    printf("device params to GPU "); fflush(stdout);
    CUDA_CHECK( cudaMalloc(&d_output, q*sizeof(float)) );
    timer.restart();
    Params params;
    Params *device_params;

    params.handle = state.gas_handle;
    params.min = 0;
    params.max = 100000000;
    params.output = d_output;
    if (alg != ALG_GPU_RTX_BLOCKS) {
        params.query = transform_querys(alg, dquery, q, n);
        params.iquery = nullptr;
    } else {
        params.query = nullptr;
        params.iquery = dquery;
        int num_blocks = (orig_n + bs - 1) / bs;
        params.num_blocks = ceil(sqrt(num_blocks + 1));
        params.block_size = bs;
    }
    printf("(%7.3f MB).........", (double)sizeof(Params)/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());

    // 5) Computation
    printf("%sComputing RMQs (%-11s)..............%s", AC_BOLDCYAN, algStr[alg], AC_RESET); fflush(stdout);
    //printf("\n");
    timer.restart();
    for (int i = 0; i < REPS; ++i) {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, q, 1, 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float timems = timer.get_elapsed_ms();
    CUDA_CHECK( cudaMemcpy(output, d_output, q*sizeof(float), cudaMemcpyDeviceToHost) );
    float time_it = timems/REPS;
    printf(AC_BOLDCYAN "done (%i reps): %f secs: [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, REPS, timems/1000.0, (double)q/(time_it/1000.0), (double)time_it*1e6/q);
    write_results(timems, q);
        
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
