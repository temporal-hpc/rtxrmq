#pragma once
float* rtx_rmq(int n, int q, float *darray, int2 *dquery, curandState *devStates) {
    Timer timer;
    float *output, *d_output;
    //float cpuMin=-1.0f;
    output = (float*)malloc(q*sizeof(float));

    // 1) Generate geometry from device data
    printf("Generating geometry......................."); fflush(stdout);
    timer.restart();
    float3* devVertices = gen_vertices_dev(n, darray);
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
    createGroupsClosestHit(state);
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
    params.handle = state.gas_handle;
    params.min = 0;
    params.max = 100000000;
    params.scale = n;
    params.output = d_output;
    params.query = dquery;
    Params *device_params;
    printf("(%7.3f MB).........", (double)sizeof(Params)/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());

    // 5) Computation
    printf("%sComputing RMQs (%-11s)..............%s", AC_BOLDCYAN, algStr[ALG_GPU_RTX], AC_RESET); fflush(stdout);
    //printf("\n");
    timer.restart();
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, q, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float timems = timer.get_elapsed_ms();
    CUDA_CHECK( cudaMemcpy(output, d_output, q*sizeof(float), cudaMemcpyDeviceToHost) );
    printf(AC_BOLDCYAN "done: %f secs: [%.2f RMQs/sec, %.3f usec/RMQ]\n" AC_RESET, timems/1000.0, (double)q/(timems/1000.0), (double)timems*1000.0/q);
    if (q < 50) {
        printf("rtx:  ");
        for (int i = 0; i < q; ++i)
            printf("%f  ", output[i]);
        printf("\ncpu:  ");
        for (int i = 0; i < q; ++i)
            printf("%f  ", cpurmq_vertex(3*n, devVertices, q, dquery, i)); // TODO copy data only once
        printf("\n");
    }
        
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
