/*!
  \file   complexD.cuh
  \brief  Complex number Operations double precision.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef COMPLEXD_CUH
#define COMPLEXD_CUH
#include "common.cuh"
#include <math.h>
typedef double2 ComplexD;

// Complex addition
static __device__ __host__ inline ComplexD ComplexAdd(ComplexD a, ComplexD b) {
  ComplexD c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline ComplexD ComplexScale(ComplexD a, double s) {
  ComplexD c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline ComplexD ComplexMul(ComplexD a, ComplexD b) {
  ComplexD c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
__device__ __forceinline__ ComplexD my_cexpf(ComplexD z) {

  ComplexD res;

  float t = exp(z.x);

  sincos(z.y, &res.y, &res.x);

  res.x *= t;

  res.y *= t;

  return res;
}

#endif
