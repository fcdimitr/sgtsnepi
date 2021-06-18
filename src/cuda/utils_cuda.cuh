/*!
  \file   utils_cuda.cuh
  \brief  Cufa utils.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef UTILS_CUDA_CUH
#define UTILS_CUDA_CUH
extern int Blocks;
extern int Threads;
template <typename T>
__global__ void initKernel(T *devPtr, const T val, const size_t nwords) {
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tidx < nwords; tidx += stride)
    devPtr[tidx] = val;
}
template <class dataPoint>
__global__ void addScalar(dataPoint *a, dataPoint scalar, uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] += scalar;
  }
}
template <class dataPoint>
__global__ void ArrayScale(volatile dataPoint *__restrict__ a,
                           const dataPoint scalar, const uint32_t length) {
  for (register int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] *= scalar;
  }
}
template <class dataPoint>
__global__ void ArrayCopy(const dataPoint *const a,
                          volatile dataPoint *__restrict__ b,
                          const uint32_t n) {
  register uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  register uint32_t stride = blockDim.x * gridDim.x;
  for (; tidx < n; tidx += stride) {
    b[tidx] = a[tidx];
  }
}

template <class dataPoint>
__global__ void copydataKernel(volatile dataPoint *__restrict__ a,
                               const dataPoint *const b,
                               const uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i];
  }
}
template <class dataPoint>
__global__ void normalize(volatile dataPoint *__restrict__ P,
                          const dataPoint sum, const uint32_t nnz) {
  for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz;
       i += gridDim.x * blockDim.x) {
    P[i] /= sum;
  }
}
inline __host__ __device__ int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
template <class dataPoint>
__global__ void addScalarToCoord(volatile dataPoint *__restrict__ y,
                                 const dataPoint scalar, const int n,
                                 const int coordinate, const int d) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    y[TID * d + coordinate] += scalar;
  }
}
#define FULL_WARP_MASK 0xFFFFFFFF
template <class T> __device__ T warp_reduce(T val) {
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
  }
  return val;
}

template <class dataType1, class dataType2>
__global__ void copymixed(dataType1 *a, dataType2 *b, const int n) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    a[TID] = (dataType1)b[TID];
  }
}
template <class dataType>
__global__ void copyCoord(dataType *a, dataType *b, const int n, const int d,
                          const int j) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    a[TID] = b[TID * d + j];
  }
}

#include <thrust/device_vector.h>

template <class dataPoint>
void centerPoints(dataPoint *y, const int n, const int d) {
  thrust::device_vector<dataPoint> x(n);
  dataPoint miny[4];

  for (int j = 0; j < d; j++) {
    copyCoord<<<Blocks, Threads>>>(thrust::raw_pointer_cast(x.data()), y, n, d,
                                   j);
    miny[j] = thrust::reduce(x.begin(), x.end(),
                             std::numeric_limits<dataPoint>::max(),
                             thrust::minimum<dataPoint>());
    addScalarToCoord<<<Blocks, Threads>>>(y, -miny[j], n, j, d);
  }
}
#endif
