#pragma once

#define COMPACT 0
//#define DEBUG
#ifdef DEBUG
#define dbg(msg) do {printf(msg "\n"); fflush(stdout);} while (0)
#else
#define dbg(msg)
#endif

template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

void launch_cuda(int n, float a, float *x, float *y);

std::string loadPtx(std::string filename) {
  std::ifstream ptx_in(filename);
  return std::string((std::istreambuf_iterator<char>(ptx_in)), std::istreambuf_iterator<char>());
}

struct Params {
  OptixTraversableHandle handle;
  float *output;
  float2 *query;
  int2 *iquery;
  float *LUP;
  int *idx_output;
  float min;
  float max;
  int num_blocks;
  int block_size;
  int nb;
};

struct GASstate {
  OptixDeviceContext context = 0;

  size_t temp_buffer_size = 0;
  CUdeviceptr d_temp_buffer = 0;
  CUdeviceptr d_temp_vertices = 0;
  CUdeviceptr d_temp_triangles = 0;
  CUdeviceptr d_instances = 0;
  float3** block_vertices;
  uint3** block_triangles;
  CUdeviceptr* d_block_vertices;
  CUdeviceptr* d_block_triangles;

  unsigned int triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;

  OptixBuildInput triangle_input = {};
  OptixBuildInput ias_instance_input = {};
  OptixTraversableHandle gas_handle;
  OptixTraversableHandle* handles;
  CUdeviceptr d_gas_output_buffer;
  size_t gas_output_buffer_size = 0;

  OptixModule ptx_module = 0;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = 0;

  OptixProgramGroup program_groups[3];
  OptixShaderBindingTable sbt = {};

  unsigned int gas_build_options;
};

void createOptixContext(GASstate &state) {
  CUDA_CHECK( cudaFree(0) ); // creates a CUDA context if there isn't already one
  OPTIX_CHECK (optixInit() ); // loads the optix library and populates the function table

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
#ifdef DEBUG
  options.logCallbackLevel = 4;
  options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
  options.logCallbackLevel = 1;
#endif 

  OptixDeviceContext optix_context = nullptr;
  optixDeviceContextCreate(0, // use current CUDA context
                           &options, &optix_context);

  state.context = optix_context;
}

// load ptx and create module
void loadAppModule(GASstate &state, CmdArgs args) {
  std::string ptx = loadPtx(BUILD_DIR "/ptx/rtx_kernels.ptx");

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef DEBUG
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

  state.pipeline_compile_options.usesMotionBlur = false;
  if (args.alg != ALG_GPU_RTX_IAS && args.alg != ALG_GPU_RTX_IAS_TRANS) {
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  } else {
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    //state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  }
  state.pipeline_compile_options.numPayloadValues = 1;
  state.pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  //OPTIX_CHECK(optixModuleCreateFromPTX(state.context, &module_compile_options, &state.pipeline_compile_options, ptx.c_str(), ptx.size(), nullptr, nullptr, &state.ptx_module));
  OPTIX_CHECK(optixModuleCreate(state.context, &module_compile_options, &state.pipeline_compile_options, ptx.c_str(), ptx.size(), nullptr, nullptr, &state.ptx_module));
}

void createProgramGroups(GASstate &state, int alg) {
  const char *rg, *ch;
  switch(alg) {
    case ALG_GPU_RTX_BLOCKS:
    case ALG_GPU_RTX_IAS:
    case ALG_GPU_RTX_IAS_TRANS:
      rg = "__raygen__rmq_blocks";
      ch = "__closesthit__rmq_blocks";
      break;
    case ALG_GPU_RTX_LUP:
      rg = "__raygen__rmq_lup";
      ch = "__closesthit__rmq_blocks";
      break;
    case ALG_GPU_RTX_BLOCKS_IDX:
      rg = "__raygen__rmq_blocks_idx";
      ch = "__closesthit__rmq_blocks_idx";
      break;
    case ALG_GPU_RTX_CAST_IDX:
      printf("Using cast rtx idx shaders\n");
      rg = "__raygen__rmq_idx";
      ch = "__closesthit__rmq_idx";
      break;
    default:
      rg = "__raygen__rmq";
      ch = "__closesthit__rmq";
      break;
  }

  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = rg;

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rmq";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = ch;
  prog_group_desc[2].hitgroup.moduleAH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}

void createGroupsClosestHit(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rmq";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rmq";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__rmq";
  prog_group_desc[2].hitgroup.moduleAH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}

void createGroupsClosestHit_Blocks(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rmq_blocks";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rmq";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__rmq_blocks";
  //prog_group_desc[2].hitgroup.moduleAH = nullptr;
  //prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;
  prog_group_desc[2].hitgroup.moduleAH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__rmq";

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}

void createGroupsClosestHit_LUP(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rmq_lup";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rmq";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__rmq_blocks";
  prog_group_desc[2].hitgroup.moduleAH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}

void createPipeline(GASstate &state) {
  OptixPipelineLinkOptions pipeline_link_options = {};
  //pipeline_link_options.maxTraceDepth = 1;
  pipeline_link_options.maxTraceDepth = 2;
//#ifdef DEBUG
//  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//#endif
  OPTIX_CHECK(optixPipelineCreate(state.context, &state.pipeline_compile_options, &pipeline_link_options, state.program_groups, 3, nullptr, nullptr, &state.pipeline));
}

void populateSBT(GASstate &state) {
  char *device_records;
  CUDA_CHECK(cudaMalloc(&device_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE));

  char *raygen_record = device_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *miss_record = device_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *hitgroup_record = device_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  char sbt_records[3 * OPTIX_SBT_RECORD_HEADER_SIZE];
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[0], sbt_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[1], sbt_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[2], sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE));

  CUDA_CHECK(cudaMemcpy(device_records, sbt_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);

  state.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
  state.sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.missRecordCount = 1;

  state.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
  state.sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.hitgroupRecordCount = 1;
}

void buildASFromDeviceData(VBHMem &mem, GASstate &state, int nverts, int ntris, float3 *devVertices, uint3 *devTriangles) {

  //const size_t vertices_size = sizeof(float3) * vertices.size();
  //CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&state.d_temp_vertices), vertices_size) );
  //CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(state.d_temp_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice) );
  state.d_temp_vertices = reinterpret_cast<CUdeviceptr>(devVertices);

  //uint3* d_triangles;
  //const size_t triangles_size = sizeof(uint3) * triangles.size();
  //CUDA_CHECK( cudaMalloc(&d_triangles, triangles_size) );
  //CUDA_CHECK( cudaMemcpy(d_triangles, triangles.data(), triangles_size, cudaMemcpyHostToDevice) );

  state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  //state.triangle_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
  state.triangle_input.triangleArray.numVertices = static_cast<unsigned int>(nverts);
  state.triangle_input.triangleArray.vertexBuffers = &state.d_temp_vertices;
  state.triangle_input.triangleArray.flags = &state.triangle_flags;
  state.triangle_input.triangleArray.numSbtRecords = 1;
  state.triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  state.triangle_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(ntris);
  state.triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(devTriangles);

  OptixAccelBuildOptions accel_options = {};
  //state.gas_build_options = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  //state.gas_build_options = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  if (COMPACT)
    state.gas_build_options = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  else
    state.gas_build_options = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.buildFlags = state.gas_build_options;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &accel_options, &state.triangle_input, 1, &gas_buffer_sizes) );

  //void *d_temp_buffer_gas;
  state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;

  //CUDA_CHECK( cudaMalloc(&d_temp_buffer_gas, temp_size) );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) );

  // non-compact output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8) );

  mem.out_buffer = gas_buffer_sizes.outputSizeInBytes;
  mem.temp_buffer = gas_buffer_sizes.tempSizeInBytes;

  OptixAccelEmitDesc emitProperty = {};
  if (COMPACT)
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  else
    emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

  OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &accel_options, 
        &state.triangle_input,
        1,
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
	    d_buffer_temp_output_gas_and_compacted_size,
	    gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, 1) 
  );  
 
  size_t compacted_gas_size;
  if (COMPACT) {
    CUDA_CHECK( cudaFree((void*)state.d_temp_buffer) );
    CUDA_CHECK( cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );
    printf("Memory usage (MB): output_buffer %f,  temp_buffer %f,  compacted %f\n", mem.out_buffer/1e6, mem.temp_buffer/1e6, compacted_gas_size/1e6); 
  }

  if (COMPACT && compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK( cudaMalloc((void**)&state.d_gas_output_buffer, compacted_gas_size) );

    // use handle as input and output
    OPTIX_CHECK( optixAccelCompact(state.context, 0, state.gas_handle, reinterpret_cast<CUdeviceptr>(state.d_gas_output_buffer), compacted_gas_size, &state.gas_handle));

    CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
  } else {
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
  }
  //CUDA_CHECK(cudaFree(d_vertices));
}

void buildBlockGeometry(VBHMem &mem, GASstate &state, int idx, int ntris) {
  OptixAccelBuildOptions accel_options = {};
  //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  state.gas_build_options = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.buildFlags = state.gas_build_options;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.numVertices = static_cast<unsigned int>(ntris * 3);
  triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&state.block_vertices[idx]);
  triangle_input.triangleArray.flags = &state.triangle_flags;
  triangle_input.triangleArray.numSbtRecords = 1;
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(ntris);
  //triangle_input.triangleArray.indexBuffer = state.d_temp_triangles;
  triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(state.block_triangles[idx]);

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &accel_options, &triangle_input, 1, &gas_buffer_sizes) );

  mem.out_buffer += gas_buffer_sizes.outputSizeInBytes;
  mem.temp_buffer = max(mem.temp_buffer, gas_buffer_sizes.tempSizeInBytes);
  //printf("Memory usage (GB): output_buffer %f,  temp_buffer %f\n", mem.out_buffer/1e9, mem.temp_buffer/1e9); 


  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) );

  // non-compact output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8) );

  OptixAccelEmitDesc emitProperty = {};
  //emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

  //dbg("before accelbuild");
  dbg("accel build");
  OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &accel_options, 
        &triangle_input,
        1,
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
	    d_buffer_temp_output_gas_and_compacted_size,
	    gas_buffer_sizes.outputSizeInBytes,
        state.handles + idx,
        &emitProperty, 1) 
  );  
  //dbg("after accelbuild");

  CUDA_CHECK(cudaFree((void*)state.d_temp_buffer));
}

void buildIAS(VBHMem &mem, GASstate &state, int nverts, int ntris, float3 *devVertices, uint3 *devTriangles, int bs, int nb, int alg) {
  //dbg("start build AS");
  state.d_temp_vertices = reinterpret_cast<CUdeviceptr>(devVertices);
  state.d_temp_triangles = reinterpret_cast<CUdeviceptr>(devTriangles);

  state.handles = (OptixTraversableHandle*)malloc(sizeof(OptixTraversableHandle)*(nb+1));
  state.block_vertices = (float3**)malloc(sizeof(float3*)*(nb+1));
  state.block_triangles = (uint3**)malloc(sizeof(uint3*)*(nb+1));
  //CUDA_CHECK(cudaMalloc(&state.d_block_vertices, sizeof(float3*)*(nb+1)));
  //CUDA_CHECK(cudaMalloc(&state.d_block_triangles, sizeof(uint3*)*(nb+1)));
  //dbg("after d_blokc mallocs");
  //state.block_vertices[0] = reinterpret_cast<CUdeviceptr>(state.d_temp_vertices);
  //state.block_triangles[0] = reinterpret_cast<CUdeviceptr>(devTriangles);
  ////dbg("afte block 0");
  //for (int i = 1; i < nb; ++i) {
  //  state.block_vertices[i] = reinterpret_cast<CUdeviceptr>(state.d_temp_vertices + (nb+bs*(i-1))*3);
  //  state.block_triangles[i] = reinterpret_cast<CUdeviceptr>(devTriangles + nb+bs*(i-1));
  //}
  ////dbg("after intermediate block");
  int idx_last = nb + bs*(nb-1);
  //printf("bs: %i,  nb: %i,  idx_last: %i, ntris: %i\n", bs, nb, idx_last, ntris); 
  //state.block_vertices[nb] = reinterpret_cast<CUdeviceptr>(state.d_temp_vertices + idx_last*3);
  //state.block_triangles[nb] = reinterpret_cast<CUdeviceptr>(devTriangles + idx_last);
  //dbg("after creating cudeviceptr");

  //printf("vertices address: ");
  //for (int i = 0; i <= nb; ++i)
  //  printf("%x ", state.block_vertices[i]);
  //printf("\n");
  //printf("triangle address: ");
  //for (int i = 0; i <= nb; ++i)
  //  printf("%x ", state.block_triangles[i]);
  //printf("\n");

  //dbg("malloc first block");
  //// geometry for minimums per block
  CUDA_CHECK(cudaMalloc((void**)&state.block_vertices[0], nb*sizeof(float3)*3));
  dbg("copy first block");
  CUDA_CHECK(cudaMemcpy((void*)state.block_vertices[0], devVertices, nb*sizeof(float3)*3, cudaMemcpyDeviceToDevice));
  dbg("malloc tringles");
  CUDA_CHECK(cudaMalloc((void**)&state.block_triangles[0], nb*sizeof(uint3)));
  CUDA_CHECK(cudaMemcpy((void*)state.block_triangles[0], devTriangles, nb*sizeof(uint3), cudaMemcpyDeviceToDevice));
  //dbg("after geometry block 1");

  // geomtery for each block
  dbg("malloc intermediate blocks");
  for (int i = 1; i < nb; ++i) {
    CUDA_CHECK(cudaMalloc((void**)&state.block_vertices[i], bs*sizeof(float3)*3));
    CUDA_CHECK(cudaMemcpy((void*)state.block_vertices[i], devVertices + nb*3 + (i-1)*bs*3,
          bs*sizeof(float3)*3, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMalloc((void**)&state.block_triangles[i], bs*sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy((void*)state.block_triangles[i], devTriangles,
          bs*sizeof(uint3), cudaMemcpyDeviceToDevice));
  }
  dbg("after geometry block 2");

  // geometry for last block
  //dbg("malloc last block");
  int size_last = ntris - idx_last;
  //printf("idx_last: %d, nb: %i\n", idx_last, nb);
  CUDA_CHECK(cudaMalloc((void**)&state.block_vertices[nb], size_last*sizeof(float3)*3));
  CUDA_CHECK(cudaMemcpy((void*)state.block_vertices[nb], devVertices + idx_last*3,
        size_last*sizeof(float3)*3, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMalloc((void**)&state.block_triangles[nb], size_last*sizeof(uint3)));
  CUDA_CHECK(cudaMemcpy((void*)state.block_triangles[nb], devTriangles,
        size_last*sizeof(uint3), cudaMemcpyDeviceToDevice));
  dbg("after geometry block 3");

  CUDA_CHECK(cudaFree(devVertices));
  CUDA_CHECK(cudaFree(devTriangles));

  //dbg("copy cudevptr to host");
  //CUDA_CHECK(cudaMemcpy((void*)state.d_block_vertices, (void*)state.block_vertices, sizeof(CUdeviceptr)*(nb+1), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy((void*)state.d_block_triangles, (void*)state.block_triangles, sizeof(CUdeviceptr)*(nb+1), cudaMemcpyHostToDevice));
  //dbg("build geometry");
  
#ifdef DEBUG
  print_vertices_dev(nb, state.block_vertices[0]);
  print_vertices_dev(bs, state.block_vertices[1]);
  print_vertices_dev(size_last, state.block_vertices[nb]);

  print_triangles_dev(nb, state.block_triangles[0]);
  print_triangles_dev(bs, state.block_triangles[1]);
  print_triangles_dev(size_last, state.block_triangles[nb]);
#endif

  cudaDeviceSynchronize();
  buildBlockGeometry(mem, state, 0, nb);
  for (int i = 1; i < nb; ++i)
    buildBlockGeometry(mem, state, i, bs);
  buildBlockGeometry(mem, state, nb, ntris-idx_last);
  cudaDeviceSynchronize();

  // IAS
  OptixInstance instance = { { 1, 0, 0, 0, 
	  		       0, 1, 0, 0, 
			       0, 0, 1, 0 } };
  // 1 0 0 0
  // 0 1 0 0
  // 0 0 1 0
  dbg("start ias");
  int x,y;
  int n_blocks = ceil(sqrt((double)nb+1));
  OptixInstance* instances = (OptixInstance*)malloc(sizeof(OptixInstance)*(nb+1));
  if(alg == ALG_GPU_RTX_IAS){
      //printf("USING GLOBAL COORDS");
      for (int i = 0; i <= nb; ++i) {
        instances[i].instanceId = i;
        instances[i].sbtOffset = 0;
        instances[i].visibilityMask = 255;
        //instances[i].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].traversableHandle = state.handles[i];
        memcpy(instances[i].transform, instance.transform, sizeof(float) * 12);
      }
  }
  else if(alg == ALG_GPU_RTX_IAS_TRANS){
      //printf("USING BLOCK LOCAL COORDS + MATRIX TRANSFORMS");
      for (int i = 0; i <= nb; ++i) {
        instances[i].instanceId = i;
        instances[i].sbtOffset = 0;
        instances[i].visibilityMask = 255;
        //instances[i].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].traversableHandle = state.handles[i];
        x = i % n_blocks;
        y = i / n_blocks;
	OptixInstance inst;
	if(i == 0){
  		inst = {{ 1, 0, 0, 0, 
	  		   0, 1, 0, 0, 
			   0, 0, 1, 0 } };
	}
	else{
        	inst = {{1.0f,    0,    0,      0,
                            0, 1.0f,    0, 2.0f*x, 
                            0,    0, 1.0f, 2.0f*y}};
	}
        memcpy(instances[i].transform, inst.transform, sizeof(float) * 12);
      }
  }
  //dbg("before copying instances");
  size_t instances_size_in_bytes = sizeof( OptixInstance ) * (nb+1);
  CUDA_CHECK( cudaMalloc( ( void** )&state.d_instances, instances_size_in_bytes ) );
  //dbg("asd");
  OptixInstance* d_inst;
  CUDA_CHECK( cudaMalloc(&d_inst, instances_size_in_bytes ) );
  CUDA_CHECK( cudaMemcpy(d_inst, instances, instances_size_in_bytes, cudaMemcpyHostToDevice ) );
  //dbg("asd1");
  CUDA_CHECK( cudaMemcpy( ( void* )state.d_instances, instances, instances_size_in_bytes, cudaMemcpyHostToDevice ) );
  //dbg("after copying instances");

  state.ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  state.ias_instance_input.instanceArray.instances = state.d_instances;
  state.ias_instance_input.instanceArray.numInstances = nb+1;

  OptixAccelBuildOptions ias_accel_options = {};
  ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  ias_accel_options.motionOptions.numKeys = 1;
  ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes ias_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &ias_accel_options, &state.ias_instance_input, 1, &ias_buffer_sizes ) );

  mem.out_buffer += ias_buffer_sizes.outputSizeInBytes;
  mem.temp_buffer = max(mem.temp_buffer, ias_buffer_sizes.tempSizeInBytes);
  //printf("Memory usage (GB): output_buffer %f,  temp_buffer %f\n", out_size/1e9, temp_size/1e9); 

  // non-compacted output
  CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
  size_t compactedSizeOffset = roundUp<size_t>( ias_buffer_sizes.outputSizeInBytes, 8ull );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_ias_and_compacted_size ), compactedSizeOffset + 8 ) );
 
  CUdeviceptr d_ias_temp_buffer;
  bool needIASTempBuffer = ias_buffer_sizes.tempSizeInBytes > state.temp_buffer_size;
  if( needIASTempBuffer )
  {
    CUDA_CHECK( cudaMalloc( (void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes ) );
  }
  else
  {
    d_ias_temp_buffer = state.d_temp_buffer;
  }

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset );

  OPTIX_CHECK( optixAccelBuild( state.context, 0, &ias_accel_options, &state.ias_instance_input, 1, d_ias_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes, &state.gas_handle, &emitProperty, 1 ) );

  if( needIASTempBuffer ) {
    CUDA_CHECK( cudaFree( (void*)d_ias_temp_buffer ) );
  } else {
    CUDA_CHECK( cudaFree( (void*)state.d_temp_buffer) );
  }
  //CUDA_CHECK( cudaFree((void*)d_buffer_temp_output_ias_and_compacted_size) );

  //state.d_ias_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
  //state.ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;
}

void updateASFromDevice(GASstate &state) {
    OptixAccelBuildOptions gas_accel_options = {};
    gas_accel_options.buildFlags = state.gas_build_options;
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,
        &gas_accel_options,
        &state.triangle_input,
        1,
        state.d_temp_buffer,
        //state.d_temp_vertices,
        state.temp_buffer_size,
        state.d_gas_output_buffer,
        state.gas_output_buffer_size,
        &state.gas_handle,
        nullptr,
        0)
    );  
}
