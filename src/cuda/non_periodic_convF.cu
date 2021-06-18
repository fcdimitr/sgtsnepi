/*!
  \file   non_periodic_convF.cu
  \brief  Implementation of the G2G computation single precision.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#include "../matrix_indexing.hpp"
#include "non_periodic_convF.cuh"
#include "utils_cuda.cuh"

#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx3(i, j, k, d1, d2) (SUB2IND3D(i, j, k, d1, d2))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))
#define CUDART_PI_F acos(-1.0)
extern int Blocks;
extern int Threads;

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(ComplexF *a,
                                                   const ComplexF *b, int size,
                                                   uint32_t nVec) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = 0; j < nVec; j++) {

    for (int i = threadID; i < size; i += numThreads) {
      a[i + j * size] = ComplexScale(ComplexMul(a[i + j * size], b[i]), 1.0f);
    }
  }
}

__global__ void setDataFft1D(ComplexF *Kc, ComplexF *Xc, int ng, int nVec,
                             float *VGrid, float hsq, int sign) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Kc[i].x = kernel1d(hsq, i);
    Kc[i].y = 0;
    if (i > 0) {
      Kc[i].x = Kc[i].x + sign * kernel1d(hsq, ng - i);
      if (sign == -1) {

        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * ng);
        Kc[i] = ComplexMul(Kc[i], my_cexpf(arg));
      }
    }
    for (int j = 0; j < nVec; j++) {
      Xc[i + j * ng].x = VGrid[i + j * ng];
      Xc[i + j * ng].y = 0;
      if (sign == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * ng);
        Xc[i + j * ng] = ComplexMul(Xc[i + j * ng], my_cexpf(arg));
      }
    }
  }
}

__global__ void setDataFft2D(ComplexF *Kc, ComplexF *Xc, int n1, int n2,
                             int nVec, const float *const VGrid, float hsq,
                             int signx, int signy) {

  register int i, j;
  for (register uint32_t TID = blockIdx.x * blockDim.x + threadIdx.x;
       TID < n1 * n2; TID += blockDim.x * gridDim.x) {
    i = TID % n1;
    j = (TID / n1);

    Kc[idx2(i, j, n1)].x = kernel2d(hsq, i, j);
    Kc[idx2(i, j, n1)].y = 0;
    if (i > 0) {
      Kc[idx2(i, j, n1)].x += signx * kernel2d(hsq, n1 - i, j);
    }
    if (j > 0) {
      Kc[idx2(i, j, n1)].x += signy * kernel2d(hsq, i, n2 - j);
    }
    if (i > 0 && j > 0) {
      Kc[idx2(i, j, n1)].x += signx * signy * kernel2d(hsq, n1 - i, n2 - j);
    }

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      Xc[idx3(i, j, iVec, n1, n2)].x = VGrid[idx3(i, j, iVec, n1, n2)];
      Xc[idx3(i, j, iVec, n1, n2)].y = 0;
      if (signx == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * n1);
        Xc[idx3(i, j, iVec, n1, n2)] =
            ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
      }
      if (signy == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * j / (2 * n2);
        Xc[idx3(i, j, iVec, n1, n2)] =
            ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
      }
    }
    if (signx == -1) {
      ComplexF arg;
      arg.x = 0;
      arg.y = -2 * CUDART_PI_F * i / (2 * n1);
      Kc[idx2(i, j, n1)] = ComplexMul(Kc[idx2(i, j, n1)], my_cexpf(arg));
    }

    if (signy == -1) {
      ComplexF arg;
      arg.x = 0;
      arg.y = -2 * CUDART_PI_F * j / (2 * n2);
      Kc[idx2(i, j, n1)] = ComplexMul(Kc[idx2(i, j, n1)], my_cexpf(arg));
    }
  }
}
__global__ void addToPhiGrid(ComplexF *Xc, float *PhiGrid, int ng,
                             float scale) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    PhiGrid[i] += scale * Xc[i].x;
  }
}

__global__ void normalizeInverse(ComplexF *Xc, int ng, uint32_t nVec) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      ComplexF arg;
      arg.x = 0;
      arg.y = +2 * CUDART_PI_F * i / (2 * ng);
      Xc[i + iVec * ng] = ComplexMul(Xc[i + iVec * ng], my_cexpf(arg));
    }
  }
}

__global__ void normalizeInverse2D(ComplexF *Xc, uint32_t n1, uint32_t n2,
                                   uint32_t nVec, int signx, int signy) {

  register int i, j;
  for (register uint32_t TID = blockIdx.x * blockDim.x + threadIdx.x;
       TID < n1 * n2; TID += blockDim.x * gridDim.x) {
    i = TID % n1;
    j = (TID / n1);

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      if (signx == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = +2 * CUDART_PI_F * i / (2 * n1);
        Xc[idx3(i, j, iVec, n1, n2)] =
            ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
      }
      if (signy == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = +2 * CUDART_PI_F * j / (2 * n2);
        Xc[idx3(i, j, iVec, n1, n2)] =
            ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
      }
    }
  }
}

void conv1dnopadcuda(float *PhiGrid, float *VGrid, float h,
                     uint32_t *const nGridDims, uint32_t nVec, int nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs) {

  uint32_t n1 = nGridDims[0];
  float hsq = h * h;
  ComplexF *Kc, *Xc;
  gpuErrchk(cudaMallocManaged(&Kc, n1 * sizeof(ComplexF)));
  gpuErrchk(cudaMallocManaged(&Xc, nVec * n1 * sizeof(ComplexF)));
  /*even*/
  setDataFft1D<<<Blocks, Threads>>>(Kc, Xc, n1, nVec, VGrid, hsq,
                                                  1);

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc, n1,
                                                                 nVec);

  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * nVec,
                                                  (0.5 / n1));

  setDataFft1D<<<Blocks, Threads>>>(Kc, Xc, n1, nVec, VGrid, hsq,
                                                  -1);

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc, n1,
                                                                 nVec);

  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);

  normalizeInverse<<<Blocks, Threads>>>(Xc, n1, nVec);

  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * nVec,
                                                  (0.5 / n1));
  gpuErrchk(cudaFree(Kc));
  gpuErrchk(cudaFree(Xc));
  return;
}
void conv2dnopadcuda(float *const PhiGrid, const float *const VGrid,
                     const float h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs) {
  float hsq = h * h;

  uint32_t n1 = nGridDims[0];
  uint32_t n2 = nGridDims[1];
  ComplexF *Kc, *Xc;
  gpuErrchk(cudaMallocManaged(&Kc, n1 * n2 * sizeof(ComplexF)));
  gpuErrchk(cudaMallocManaged(&Xc, nVec * n1 * n2 * sizeof(ComplexF)));

  // ============================== EVEN-EVEN

  setDataFft2D<<<Blocks, Threads>>>(Kc, Xc, n1, n2, nVec, VGrid,
                                                  hsq, 1, 1);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);
  // cudaDeviceSynchronize();

  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc,
                                                                 n1 * n2, nVec);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  // cudaDeviceSynchronize();
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * n2 * nVec,
                                                  (0.25 / (n1 * n2)));
  // cudaDeviceSynchronize();

  // ============================== ODD-EVEN

  setDataFft2D<<<Blocks, Threads>>>(Kc, Xc, n1, n2, nVec, VGrid,
                                                  hsq, -1, 1);
  // cudaDeviceSynchronize();

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);
  // cudaDeviceSynchronize();
  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc,
                                                                 n1 * n2, nVec);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  // cudaDeviceSynchronize();
  normalizeInverse2D<<<Blocks, Threads>>>(Xc, n1, n2, nVec, -1,
                                                        1);
  // cudaDeviceSynchronize();
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * n2 * nVec,
                                                  (0.25 / (n1 * n2)));
  // cudaDeviceSynchronize();
  // ============================== EVEN-ODD

  setDataFft2D<<<Blocks, Threads>>>(Kc, Xc, n1, n2, nVec, VGrid,
                                                  hsq, 1, -1);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);
  // cudaDeviceSynchronize();
  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc,
                                                                 n1 * n2, nVec);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  // cudaDeviceSynchronize();
  normalizeInverse2D<<<Blocks, Threads>>>(Xc, n1, n2, nVec, 1,
                                                        -1);
  // cudaDeviceSynchronize();
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * n2 * nVec,
                                                  (0.25 / (n1 * n2)));
  // cudaDeviceSynchronize();
  // ============================== ODD-ODD

  setDataFft2D<<<Blocks, Threads>>>(Kc, Xc, n1, n2, nVec, VGrid,
                                                  hsq, -1, -1);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);
  // cudaDeviceSynchronize();
  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc,
                                                                 n1 * n2, nVec);
  // cudaDeviceSynchronize();
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  // cudaDeviceSynchronize();
  normalizeInverse2D<<<Blocks, Threads>>>(Xc, n1, n2, nVec, -1,
                                                        -1);
  // cudaDeviceSynchronize();
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * n2 * nVec,
                                                  (0.25 / (n1 * n2)));
  // cudaDeviceSynchronize();
  gpuErrchk(cudaFree(Kc));
  gpuErrchk(cudaFree(Xc));
  return;
}

__global__ void setDataFft3D(ComplexF *Kc, ComplexF *Xc, const int n1,
                             const int n2, const int n3, const int nVec,
                             const float *const VGrid, const float hsq,
                             const int signx, const int signy,
                             const int signz) {
  register int i, j, k;
  register ComplexF K, X;
  for (register uint32_t TID = blockIdx.x * blockDim.x + threadIdx.x;
       TID < n1 * n2 * n3; TID += blockDim.x * gridDim.x) {
    i = TID % n1;
    j = (TID / n1) % n2;
    k = (TID / n1) / n2;
    K.x = kernel3d(hsq, i, j, k);
    K.y = 0;
    if (i > 0) {
      K.x += signx * kernel3d(hsq, n1 - i, j, k);
    }
    if (j > 0) {
      K.x += signy * kernel3d(hsq, i, n2 - j, k);
    }
    if (i > 0 && j > 0) {
      K.x += signx * signy * kernel3d(hsq, n1 - i, n2 - j, k);
    }
    if (k > 0) {
      K.x += signz * kernel3d(hsq, i, j, n3 - k);
    }
    if (k > 0 && i > 0) {
      K.x += signx * signz * kernel3d(hsq, n1 - i, j, n3 - k);
    }
    if (k > 0 && j > 0) {
      K.x += signy * signz * kernel3d(hsq, i, n2 - j, n3 - k);
    }
    if (k > 0 && i > 0 && j > 0) {
      K.x += signx * signy * signz * kernel3d(hsq, n1 - i, n2 - j, n3 - k);
    }

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      X.x = VGrid[idx4(i, j, k, iVec, n1, n2, n3)];
      X.y = 0;
      if (signx == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * n1);
        X = ComplexMul(X, my_cexpf(arg));
      }
      if (signy == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * j / (2 * n2);
        X = ComplexMul(X, my_cexpf(arg));
      }
      if (signz == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * k / (2 * n3);
        X = ComplexMul(X, my_cexpf(arg));
      }
      Xc[idx4(i, j, k, iVec, n1, n2, n3)] = X;
    }
    if (signx == -1) {
      ComplexF arg;
      arg.x = 0;
      arg.y = -2 * CUDART_PI_F * i / (2 * n1);
      K = ComplexMul(K, my_cexpf(arg));
    }

    if (signy == -1) {
      ComplexF arg;
      arg.x = 0;
      arg.y = -2 * CUDART_PI_F * j / (2 * n2);
      K = ComplexMul(K, my_cexpf(arg));
    }

    if (signz == -1) {
      ComplexF arg;
      arg.x = 0;
      arg.y = -2 * CUDART_PI_F * k / (2 * n3);
      K = ComplexMul(K, my_cexpf(arg));
    }
    Kc[idx3(i, j, k, n1, n2)] = K;
  }
}

__global__ void normalizeInverse3D(ComplexF *Xc, uint32_t n1, uint32_t n2,
                                   uint32_t n3, uint32_t nVec, int signx,
                                   int signy, int signz) {

  register int i, j, k;
  for (register uint32_t TID = blockIdx.x * blockDim.x + threadIdx.x;
       TID < n1 * n2 * n3; TID += blockDim.x * gridDim.x) {
    i = TID % n1;
    j = (TID / n1) % n2;
    k = (TID / n1) / n2;

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      if (signx == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = +2 * CUDART_PI_F * i / (2 * n1);
        Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
            ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
      }
      if (signy == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = +2 * CUDART_PI_F * j / (2 * n2);
        Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
            ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
      }
      if (signz == -1) {
        ComplexF arg;
        arg.x = 0;
        arg.y = +2 * CUDART_PI_F * k / (2 * n3);
        Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
            ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
      }
    }
  }
}
void term3D(ComplexF *Kc, ComplexF *Xc, uint32_t n1, uint32_t n2, uint32_t n3,
            uint32_t nVec, const float *const VGrid, float *PhiGrid, float hsq,
            cufftHandle plan, cufftHandle plan_rhs, int signx, int signy,
            int signz) {

  setDataFft3D<<<Blocks, Threads>>>(
      Kc, Xc, n1, n2, n3, nVec, VGrid, hsq, signx, signy, signz);

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(
      Xc, Kc, n1 * n2 * n3, nVec);

  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  if (signx == -1 || signy == -1 || signz == -1) {
    normalizeInverse3D<<<Blocks, Threads>>>(Xc, n1, n2, n3, nVec,
                                                          signx, signy, signz);
  }
  addToPhiGrid<<<Blocks, Threads>>>(
      Xc, PhiGrid, n1 * n2 * n3 * nVec, (0.125 / (n1 * n2 * n3)));
}

void conv3dnopadcuda(float *const PhiGrid, const float *const VGrid,
                     const float h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,
                     cufftHandle &plan, cufftHandle &plan_rhs) {

  float hsq = h * h;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1 = nGridDims[0];
  uint32_t n2 = nGridDims[1];
  uint32_t n3 = nGridDims[2];
  ComplexF *Kc, *Xc;
  gpuErrchk(cudaMallocManaged(&Kc, n1 * n2 * n3 * sizeof(ComplexF)));
  gpuErrchk(cudaMallocManaged(&Xc, nVec * n1 * n2 * n3 * sizeof(ComplexF)));
  // ============================== EVEN-EVEN-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, 1,
         1);

  // ============================== ODD-EVEN-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, 1,
         1);

  // ============================== EVEN-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, -1,
         1);

  // ============================== ODD-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, -1,
         1);

  // ============================== EVEN-EVEN-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, 1,
         -1);

  // ============================== EVEN-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, 1,
         -1);

  // ============================== EVEN-ODD-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, -1,
         -1);

  // ============================== ODD-ODD-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, -1,
         -1);
  gpuErrchk(cudaFree(Kc));
  gpuErrchk(cudaFree(Xc));
}
