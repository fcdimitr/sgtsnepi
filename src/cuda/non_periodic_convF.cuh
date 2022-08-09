/*!
  \file   non_periodic_convF.cuh
  \brief  Implementation of the G2G computation single precision headers.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef NUFFT_CUH
#define NUFFT_CUH
#include "common.cuh"
#include "complexF.cuh"
#include <cufft.h>
#include <cufftXt.h>

#ifndef KERNELS
#define KERNELS
__inline__ __device__ __host__ float kernel1d(float hsq, float i) {
  return pow(1.0 + hsq * i * i, -2);
}

__inline__ __device__ __host__ float kernel2d(float hsq, float i, float j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

__inline__ __device__ __host__ float kernel3d(float hsq, float i, float j,
                                              float k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif

void conv1dnopadcuda(float *PhiGrid, float *VGrid, float h,
                     uint32_t *const nGridDims, uint32_t nVec, int nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs);
void conv2dnopadcuda(float *const PhiGrid, const float *const VGrid,
                     const float h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs);
void conv3dnopadcuda(float *const PhiGrid, const float *const VGrid,
                     const float h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs);
#endif
