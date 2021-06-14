/*!
  \file   gridding.cu
  \brief  Implementation of the S2G computation headers.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef GRIDDING_CUH
#define GRIDDING_CUH
#include "common.cuh"
#define LAGRANGE_INTERPOLATION

#ifdef LAGRANGE_INTERPOLATION
template <class dataPoint>
__inline__ __host__ __device__ dataPoint l1(dataPoint d) {
  return 0.5 * d * d * d - 1.0 * d * d - 0.5 * d + 1;
}
template <class dataPoint>
__inline__ __host__ __device__ dataPoint l2(dataPoint d) {
  dataPoint cc = 1.0 / 6.0;
  return -cc * d * d * d + 1.0 * d * d - 11 * cc * d + 1;
}

#else
template <class dataPoint>
__inline__ __host__ __device__ dataPoint l1(dataPoint d) {
  return 1.5 * d * d * d - 2.5 * d * d + 1;
}
template <class dataPoint>
__inline__ __host__ __device__ dataPoint l2(dataPoint d) {
  return -0.5 * d * d * d + 2.5 * d * d - 4 * d + 2;
}

#endif

template <class dataPoint>
void s2g(dataPoint *VGrid, dataPoint *y, dataPoint *VScat, uint32_t nGridDim,
         uint32_t n, uint32_t d, uint32_t m);
template <class dataPoint>
void g2s(dataPoint *PhiScat, dataPoint *PhiGrid, dataPoint *y,
         uint32_t nGridDim, uint32_t n, uint32_t d, uint32_t m);
template <class dataPoint>
__global__ void g2s1d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec);
template <class dataPoint>
__global__ void g2s2d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec);
template <class dataPoint>
__global__ void g2s3d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec);

template <class dataPoint, class sumType>
__global__ void s2g1d(sumType *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
template <class dataPoint, class sumType>
__global__ void s2g2d(sumType *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
template <class dataPoint, class sumType>
__global__ void s2g3d(sumType *__restrict__ V, dataPoint *y, dataPoint *q,
                      uint32_t ng, uint32_t nPts, uint32_t nDim, uint32_t nVec);

#endif
