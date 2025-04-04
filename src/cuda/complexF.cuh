/*!
  \file   complexF.cuh
  \brief  Complex number Operations single precision.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef COMPLEXF_CUH
#define COMPLEXF_CUH
#include "common.cuh"
#include <math.h>
typedef float2 ComplexF;

// Complex addition
static __device__ __host__ inline ComplexF ComplexAdd(ComplexF a, ComplexF b) {
  ComplexF c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline ComplexF ComplexScale(ComplexF a, float s) {
  ComplexF c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline ComplexF ComplexMul(ComplexF a, ComplexF b) {
  ComplexF c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
__device__ __forceinline__ ComplexF my_cexpf(ComplexF z) {

  ComplexF res;

  float t = expf(z.x);

  sincosf(z.y, &res.y, &res.x);

  res.x *= t;

  res.y *= t;

  return res;
}

#endif
