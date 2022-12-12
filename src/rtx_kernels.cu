#include <optix.h>
#include <math.h>

struct Params {
  OptixTraversableHandle handle;
  float *output;
  float2 *query;
  int2 *iquery;
  float min;
  float max;
  int num_blocks;
  int block_size;
};

extern "C" static __constant__ Params params;

extern "C" __global__ void __raygen__rmq() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;

  float2 q = params.query[idx.x];
  float3 ray_origin = make_float3(min, q.x, q.y);
  float3 ray_direction = make_float3(1.0, 0.0, 0.0);
  //printf("ray %i,  (l,r)=(%f, %f)\n", idx.x, (float)q.x, (float)q.y);

  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  unsigned int payload = __float_as_uint(max);
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
      visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);

  params.output[idx.x] = __uint_as_float(payload) + min;
}

extern "C" __global__ void  __closesthit__rmq() {
  float curr_tmax = optixGetRayTmax();
  optixSetPayload_0(__float_as_uint(curr_tmax));

  //float3 v[3];
  //int Pidx = optixGetPrimitiveIndex();
  //optixGetTriangleVertexData(
  //    params.handle,
  //    Pidx,
  //    optixGetSbtGASIndex(),
  //    curr_tmax,
  //    v);
  //const uint3 idx = optixGetLaunchIndex();

  //printf("ray %i:  prim %i:\n\t%f %f %f\n\t%f %f %f\n\t%f %f %f\n", idx.x, Pidx,
  //    v[0].x, v[0].y, v[0].z,
  //    v[1].x, v[1].y, v[1].z,
  //    v[2].x, v[2].y, v[2].z);
}

extern "C" __global__ void  __miss__rmq() {
  optixSetPayload_0(__float_as_uint(INFINITY));
}

extern "C" __global__ void __raygen__rmq_blocks() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;


  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  unsigned int payload = __float_as_uint(max);

  int2 q = params.iquery[idx.x];
  int lB = q.x / params.block_size;
  int rB = q.y / params.block_size;

  //printf("Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n", idx.x, q.x, q.y, lB, rB);
  
  if (lB == rB) {
    float x = 2*(lB+1) + (float)(q.x%params.block_size)/params.block_size;
    float y = (float)(q.y%params.block_size) / params.block_size;
    //printf("1B Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n    x = %f,  y = %f\n", idx.x, q.x, q.y, lB, rB, x, y);
    float3 ray_origin = make_float3(min, x, y);
    float3 ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);

    params.output[idx.x] = __uint_as_float(payload) + min;
    return;
  }

  // search min in fully contained blocks
  float x = (float)(lB+1) / params.num_blocks;
  float y = (float)(rB-1) / params.num_blocks;
  float3 ray_origin = make_float3(min, x, y);
  float3 ray_direction = make_float3(1.0, 0.0, 0.0);
  //printf("ray %i,  (l,r)=(%f, %f)\n", idx.x, (float)q.x, (float)q.y);
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
      visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);

  // search min in first partial block
  int mod = q.x % params.block_size;
  if (mod) {
    x = 2*(lB+1) + ((float)mod / params.block_size);
    y = 1.0;
    ray_origin = make_float3(min, x, y);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);
  }

  // search min in last partial block
  mod = q.y % params.block_size;
  if (mod != params.block_size-1) {
    x = 2*(rB+1);
    y = (float)mod / params.block_size;
    //printf("3R Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n    x = %f,  y = %f\n", idx.x, q.x, q.y, lB, rB, x, y);
    ray_origin = make_float3(min, x, y);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);
  }

  params.output[idx.x] = __uint_as_float(payload) + min;
}

extern "C" __global__ void  __closesthit__rmq_blocks() {
  float curr_tmax = optixGetRayTmax();
  float prev_val = __uint_as_float(optixGetPayload_0());
  float val = curr_tmax < prev_val ? curr_tmax : prev_val;
  optixSetPayload_0(__float_as_uint(val));
}
