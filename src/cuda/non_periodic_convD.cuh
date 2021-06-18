/*!
  \file   non_periodic_convD.cuh
  \brief  Implementation of the G2G computation double precision headers.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef NUFFTD_CUH
#define NUFFTD_CUH
#include "common.cuh"
#include "complexD.cuh"
#include <cufft.h>
#include <cufftXt.h>

#ifndef KERNELSD
#define KERNELSD
__inline__ __device__ __host__ double kernel1d(double hsq, double i) {
  return pow(1.0 + hsq * i * i, -2);
}

__inline__ __device__ __host__ double kernel2d(double hsq, double i, double j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

__inline__ __device__ __host__ double kernel3d(double hsq, double i, double j,
                                              double k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif

void conv1dnopadcuda(double *PhiGrid, double *VGrid, double h,
                     uint32_t *const nGridDims, uint32_t nVec, int nDim,  cufftHandle& plan,cufftHandle& plan_rhs);
void conv2dnopadcuda(double *const PhiGrid, const double *const VGrid,
                     const double h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,  cufftHandle& plan,cufftHandle& plan_rhs);
void conv3dnopadcuda(double *const PhiGrid, const double *const VGrid,
                     const double h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,  cufftHandle& plan,cufftHandle& plan_rhs);
#endif
