/*!
  \file   gridding.cu
  \brief  Implementation of the S2G computation.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#include "../matrix_indexing.hpp"
#include "gridding.cuh"
#include "utils_cuda.cuh"
extern int Blocks;
extern int Threads;
//#define MIXED_PREC_SUM
#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))
#define y(i, j) y[i * nDim + j]

#ifdef MIXED_PREC_SUM

template <class dataPoint>
void s2g(dataPoint *VGrid, dataPoint *y, dataPoint *VScat, uint32_t nGridDim,
         uint32_t n, uint32_t d, uint32_t m) {
  double *VGridD;
  int szV = pow(nGridDim + 2, d) * m;
  CUDA_CALL(cudaMallocManaged(&VGridD, szV * sizeof(double)));
  initKernel<<<Blocks, Threads>>>(VGridD, (double)0, szV);

  switch (d) {

  case 1:
    s2g1d<<<Blocks, Threads>>>(VGridD, y, VScat, nGridDim + 2, n, d, m);
    break;

  case 2:
    s2g2d<<<Blocks, Threads>>>(VGridD, y, VScat, nGridDim + 2, n, d, m);
    break;

  case 3:
    s2g3d<<<Blocks, Threads>>>(VGridD, y, VScat, nGridDim + 2, n, d, m);
    break;
  }
  copymixed<<<Blocks, Threads>>>(VGrid, VGridD, szV);
  cudaFree(VGridD);
}

#else

template <class dataPoint>
void s2g(dataPoint *VGrid, dataPoint *y, dataPoint *VScat, uint32_t nGridDim,
         uint32_t n, uint32_t d, uint32_t m) {
  switch (d) {

  case 1:
    s2g1d<<<Blocks, Threads>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);
    break;
  case 2:
    s2g2d<<<Blocks, Threads>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);
    break;

  case 3:
    s2g3d<<<Blocks, Threads>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);
    break;
  }
}

#endif

template <class dataPoint>
void g2s(dataPoint *PhiScat, dataPoint *PhiGrid, dataPoint *y,
         uint32_t nGridDim, uint32_t n, uint32_t d, uint32_t m) {
  switch (d) {

  case 1:
    g2s1d<<<Blocks, Threads>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);

    break;

  case 2:
    g2s2d<<<Blocks, Threads>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;

  case 3:
    g2s3d<<<Blocks, Threads>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;
  }
}

template <class dataPoint, class sumType>
__global__ void s2g1d(sumType *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec) {
  dataPoint v1[4];
  register uint32_t f1;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y(TID, 0));
    d = y(TID, 0) - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    for (int j = 0; j < nVec; j++) {
      dataPoint qv = q[nPts * j + TID];
      for (int idx1 = 0; idx1 < 4; idx1++) {
        atomicAdd(&V[f1 + idx1 + j * ng], (sumType)qv * v1[idx1]);
      }
    }
  }
}
template <class dataPoint>
__global__ void g2s1d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  uint32_t f1;
  dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    f1 = (uint32_t)floor(y(TID, 0));
    d = y(TID, 0) - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {
      dataPoint accum = 0;
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        accum += V[f1 + idx1 + j * ng] * v1[idx1];
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}

template <class dataPoint, class sumType>
__global__ void s2g2d(sumType *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  register uint32_t f1;
  register uint32_t f2;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floorf(y(TID, 0));
    d = y(TID, 0) - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    f2 = (uint32_t)floorf(y(TID, 1));
    d = y(TID, 1) - (dataPoint)f2;
    v2[0] = l2(1 + d);
    v2[1] = l1(d);
    v2[2] = l1(1 - d);
    v2[3] = l2(2 - d);

    for (int j = 0; j < nVec; j++) {

      for (int idx2 = 0; idx2 < 4; idx2++) {

        for (int idx1 = 0; idx1 < 4; idx1++) {

          atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng],
                    (sumType)((q[j + nVec * TID] * v2[idx2]) * v1[idx1]));
        }
      }
    }
  }
}

template <class dataPoint>
__global__ void g2s2d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  register uint32_t f1;
  register uint32_t f2;
  register dataPoint d;
  register dataPoint accum = 0;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y(TID, 0));
    d = y(TID, 0) - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    f2 = (uint32_t)floor(y(TID, 1));
    d = y(TID, 1) - (dataPoint)f2;
    v2[0] = l2(1 + d);
    v2[1] = l1(d);
    v2[2] = l1(1 - d);
    v2[3] = l2(2 - d);

    for (int j = 0; j < nVec; j++) {
      accum = 0;
      for (int idx2 = 0; idx2 < 4; idx2++) {
        dataPoint qv = v2[idx2];

        for (int idx1 = 0; idx1 < 4; idx1++) {

          accum +=
              V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng] * qv * v1[idx1];
        }
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}

template <class dataPoint, class sumType>
__global__ void s2g3d(sumType *__restrict__ V, dataPoint *y, dataPoint *q,
                      uint32_t ng, uint32_t nPts, uint32_t nDim,
                      uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  dataPoint v3[4];
  register uint32_t f1, f2, f3;
  register dataPoint d;
  register dataPoint y1, y2, y3;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    y1 = y(TID, 0);
    y2 = y(TID, 1);
    y3 = y(TID, 2);
    f1 = (uint32_t)floor(y1);
    d = y1 - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    f2 = (uint32_t)floor(y2);
    d = y2 - (dataPoint)f2;
    v2[0] = l2(1 + d);
    v2[1] = l1(d);
    v2[2] = l1(1 - d);
    v2[3] = l2(2 - d);

    f3 = (uint32_t)floor(y3);
    d = y3 - (dataPoint)f3;
    v3[0] = l2(1 + d);
    v3[1] = l1(d);
    v3[2] = l1(1 - d);
    v3[3] = l2(2 - d);

    for (int j = 0; j < 4; j++) {
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          dataPoint qv = q[j + 4 * TID] * v2[idx2] * v3[idx3];

          for (int idx1 = 0; idx1 < 4; idx1++) {
            atomicAdd(&V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng, ng, ng)],
                      (sumType)qv * v1[idx1]);
          }
        }
      }
    }
  }
}

template <class dataPoint>
__global__ void g2s3d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  dataPoint v3[4];
  register uint32_t f1;
  register uint32_t f2;
  register uint32_t f3;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y(TID, 0));
    d = y(TID, 0) - (dataPoint)f1;
    v1[0] = l2(1 + d);
    v1[1] = l1(d);
    v1[2] = l1(1 - d);
    v1[3] = l2(2 - d);

    f2 = (uint32_t)floor(y(TID, 1));
    d = y(TID, 1) - (dataPoint)f2;
    v2[0] = l2(1 + d);
    v2[1] = l1(d);
    v2[2] = l1(1 - d);
    v2[3] = l2(2 - d);

    f3 = (uint32_t)floor(y(TID, 2));
    d = y(TID, 2) - (dataPoint)f3;
    v3[0] = l2(1 + d);
    v3[1] = l1(d);
    v3[2] = l1(1 - d);
    v3[3] = l2(2 - d);

    for (int j = 0; j < nVec; j++) {
      dataPoint accum = 0;
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          dataPoint qv = v2[idx2] * v3[idx3];

          for (int idx1 = 0; idx1 < 4; idx1++) {

            accum += V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng, ng, ng)] *
                     qv * v1[idx1];
          }
        }
        Phi[TID + j * nPts] = accum;
      }
    }
  }
}
template void s2g(float *VGrid, float *y, float *VScat, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m);
template void g2s(float *PhiScat, float *PhiGrid, float *y, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m);

template void s2g(double *VGrid, double *y, double *VScat, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m);
template void g2s(double *PhiScat, double *PhiGrid, double *y,
                  uint32_t nGridDim, uint32_t n, uint32_t d, uint32_t m);
