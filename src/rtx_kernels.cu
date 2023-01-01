#include <optix.h>
#include <math.h>

//#define DBG_RAY 17780

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
  int num_blocks = params.num_blocks;
  int block_size = params.block_size;


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
  int lB = q.x / block_size;
  int rB = q.y / block_size;

    //printf("Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n", idx.x, q.x, q.y, lB, rB);
  int bx, by;
  float x, y;
  float3 ray_origin, ray_direction;
  
  if (lB == rB) {
    bx = (lB+1) % num_blocks;
    by = (rB+1) / num_blocks;
    int mx = q.x % block_size;
    int my = q.y % block_size;
    x = 2*bx + ((float)mx / block_size);
    y = 2*by + ((float)my / block_size);
#ifdef DBG_RAY
    if (idx.x == DBG_RAY)
      printf("1B Ray %i, query (%i,%i)\n    bx %i    by %i   mx %i   my %i \n    lB = %i,  rB = %i  nb = %i\n    x = %f,  y = %f\n", idx.x, q.x, q.y, bx, by, mx, my, lB, rB, num_blocks, x, y);
#endif
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);

    params.output[idx.x] = __uint_as_float(payload) + min;
    return;
  }

  // search min in fully contained blocks
  if (lB < rB-1) {
    x = (float)(lB+1) / (1<<23);
    y = (float)(rB-1) / (1<<23);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);
#ifdef DBG_RAY
    if (idx.x == DBG_RAY)
      printf("\nray %i,  (l,r)=(%i, %i)\n    bs = %i    nb = %i\n    x = %.15f,  y = %.15f\n    payload %f\n", idx.x, q.x, q.y, block_size, num_blocks, x, y, __uint_as_float(payload));
#endif
  }

  // search min in first partial block
  int mod = q.x % block_size;
  bx = (lB+1) % num_blocks;
  by = (lB+1) / num_blocks;
  x = 2*bx + ((float)mod / block_size);
  y = 2*by+1;
  ray_origin = make_float3(min, x, y);
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
      visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);
#ifdef DBG_RAY
  if (idx.x == DBG_RAY)
    printf("3R left Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n    x = %f,  y = %f\n    payload %f\n", idx.x, q.x, q.y, lB, rB, x, y, __uint_as_float(payload));
#endif

  // search min in last partial block
  mod = q.y % block_size;
  bx = (rB+1) % num_blocks;
  by = (rB+1) / num_blocks;
  x = 2*bx;
  y = 2*by + (float)mod / block_size;
  ray_origin = make_float3(min, x, y);
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
      visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);
#ifdef DBG_RAY
  if (idx.x == DBG_RAY)
    printf("3R Ray right %i, query (%i,%i)\n    lB = %i,  rB = %i\n    x = %f,  y = %f\n    payload %f\n", idx.x, q.x, q.y, lB, rB, x, y, __uint_as_float(payload));
#endif

  params.output[idx.x] = __uint_as_float(payload) + min;
}

extern "C" __global__ void  __closesthit__rmq_blocks() {
  float curr_tmax = optixGetRayTmax();
  float prev_val = __uint_as_float(optixGetPayload_0());
  float val = curr_tmax < prev_val ? curr_tmax : prev_val;
  optixSetPayload_0(__float_as_uint(val));
}
