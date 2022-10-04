#include <optix.h>
#include <math.h>

struct Params {
  OptixTraversableHandle handle;
  float *output;
  int2 *query;
  float min;
  float max;
  float scale;
};

extern "C" static __constant__ Params params;

extern "C" __global__ void __raygen__rmq() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;

  int2 q = params.query[idx.x];
  float3 ray_origin = make_float3(min, (float)q.x/params.scale, (float)q.y/params.scale);
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




