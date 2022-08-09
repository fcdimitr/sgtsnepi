/*!
  \file   Frep.cu
  \brief  Implementation for the apprpoximation of the Repulsive term.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#include "../types.hpp"
#include "Frep.cuh"
#include "gpu_timer.h"
#include "gridding.cuh"

#define N_GRID_SIZE 137
int Blocks = 64;
int Threads = 1024;
int initialization = 0;
template <class dataPoint>
__global__ void ComputeChargesKernel(volatile dataPoint *__restrict__ VScat,
                                     const dataPoint *const y, const int n,
                                     const int d) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {

    switch (d) {
    case 1: {
      VScat[TID] = 1;
      VScat[TID + n] = y[TID];
      break;
    }
    case 2: {
      VScat[3 * TID] = 1;
      VScat[3 * TID + 1] = y[2 * TID];
      VScat[3 * TID + 2] = y[2 * TID + 1];
      break;
    }
    case 3: {
      VScat[4 * TID] = 1;
      VScat[4 * TID + 1] = y[3 * TID];
      VScat[4 * TID + 2] = y[3 * TID + 1];
      VScat[4 * TID + 3] = y[3 * TID + 2];
      break;
    }
    }
  }
}
template <class dataPoint>
void ComputeCharges(dataPoint *VScat, dataPoint *y_d, const int n,
                    const int d) {
  ComputeChargesKernel<<<Blocks, Threads>>>(VScat, y_d, n, d);
}
template <class dataPoint>
__global__ void
compute_repulsive_forces_kernel(volatile dataPoint *__restrict__ frep,
                                const dataPoint *const Y, const int num_points,
                                const int nDim, const dataPoint *const Phi,
                                volatile dataPoint *__restrict__ zetaVec) {

  register dataPoint Ysq = 0;
  register dataPoint z = 0;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < num_points; TID += gridDim.x * blockDim.x) {
    Ysq = 0;
    z = 0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[nDim * TID + j] * Y[nDim * TID + j];
      z -= 2 * Y[nDim * TID + j] * Phi[TID + (num_points) * (j + 1)];
      frep[nDim * TID + j] =
          Y[nDim * TID + j] * Phi[TID] - Phi[TID + (j + 1) * num_points];
    }

    z += (1 + 2 * Ysq) * Phi[TID];
    zetaVec[TID] = z;
  }
}

template <class dataPoint>
dataPoint zetaAndForce(dataPoint *Ft_d, dataPoint *y_d, int n, int d,
                       dataPoint *Phi,
                       thrust::device_vector<dataPoint> &zetaVec) {

  compute_repulsive_forces_kernel<<<Blocks, Threads>>>(
      Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()));
  // cudaDeviceSynchronize();
  dataPoint z = thrust::reduce(zetaVec.begin(), zetaVec.end()) - n;

  normalize<<<Blocks, Threads>>>(Ft_d, z, n * d);
  return z;
}

int getGridSize(int nGrid) {

  // list of FFT sizes that work "fast" with FFTW
  int listGridSize[N_GRID_SIZE] = {
      8,   9,   10,  11,  12,  13,  14,  15,  16,  20,  25,  26,  28,  32,
      33,  35,  36,  39,  40,  42,  44,  45,  48,  49,  50,  52,  54,  55,
      56,  60,  63,  64,  65,  66,  70,  72,  75,  77,  78,  80,  84,  88,
      90,  91,  96,  98,  99,  100, 104, 105, 108, 110, 112, 117, 120, 125,
      126, 130, 132, 135, 140, 144, 147, 150, 154, 156, 160, 165, 168, 175,
      176, 180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224,
      225, 231, 234, 240, 245, 250, 252, 260, 264, 270, 273, 275, 280, 288,
      294, 297, 300, 308, 312, 315, 320, 325, 330, 336, 343, 350, 351, 352,
      360, 364, 375, 378, 385, 390, 392, 396, 400, 416, 420, 432, 440, 441,
      448, 450, 455, 462, 468, 480, 490, 495, 500, 504, 512};

  // select closest (larger) size for given grid size
  for (int i = 0; i < N_GRID_SIZE; i++)
    if ((nGrid + 2) <= listGridSize[i])
      return listGridSize[i] - 2;

  return listGridSize[N_GRID_SIZE - 1] - 2;
}

void initializeLaunchParameters(int *GridSize, int *BlockSize, int d) {

  switch (d) {
  case 1: {
    cudaOccupancyMaxPotentialBlockSize(GridSize, BlockSize, s2g1d<coord, coord>,
                                       0, 0);
    break;
  }
  case 2: {
    cudaOccupancyMaxPotentialBlockSize(GridSize, BlockSize, s2g2d<coord, coord>,
                                       0, 0);
    break;
  }
  case 3: {
    cudaOccupancyMaxPotentialBlockSize(GridSize, BlockSize, s2g3d<coord, coord>,
                                       0, 0);
    break;
  }
  }

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  std::cout << "BlockSize and minimum gridSize for potential maximal occupancy "
               "of the S2G kernel: "
            << *GridSize << " , " << *BlockSize << "\n";
  std::cout << "Number of SMs: " << numSMs << "\n";
}
coord computeFrepulsive_GPU(coord *Freph, coord *yh, int n, int d, double h,
                            double *timeInfo) {

  if (initialization == 0) {
    initializeLaunchParameters(&Blocks, &Threads, d);
    std::cout << "Choosing GridSize= " << Blocks
              << " and BlockSize= " << Threads << "\n";
    initialization = 1;
  }
  struct GpuTimer timer;
  timer.Start();

  coord *y;
  gpuErrchk(cudaMallocManaged(&y, (d)*n * sizeof(coord)));
  gpuErrchk(cudaMemcpy(y, yh, n * d * sizeof(coord), cudaMemcpyHostToDevice));
  coord *Frep;
  gpuErrchk(cudaMallocManaged(&Frep, n * d * sizeof(coord)));

  timer.Stop();

  timeInfo[4] = timer.Elapsed() / 1000.0;

  coord zeta = computeFrepulsive_gpu(Frep, y, n, d, h, timeInfo);
  timer.Start();

  gpuErrchk(
      cudaMemcpy(Freph, Frep, n * d * sizeof(coord), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(Frep));
  gpuErrchk(cudaFree(y));
  timer.Stop();

  timeInfo[4] += timer.Elapsed() / 1000.0;
  return zeta;
}

coord computeFrepulsive_gpu(coord *Frep, coord *y, int n, int d, double h,
                            double *timeInfo) {

  // timeInfo=[S2Gtime G2Gtime G2Gtime Phi2Forcetime PreparatoryOrOtherProctime]
  struct GpuTimer timer;
  timer.Start();

  int m = d + 1;

  // ~~~~~~~~~~ move data to (0,0,...)
  centerPoints(y, n, d);

  // Find interpolation grid size
  thrust::device_ptr<coord> yVec_ptr(y);
  coord maxy =
      thrust::reduce(yVec_ptr, yVec_ptr + n * d, 0.0, thrust::maximum<coord>());

  int ng = std::max((int)std::ceil(maxy / (coord)h), 14);

  int n1 = getGridSize(ng);
  int nGrid = n1;

  // Memory allocations
  coord *yt;
  gpuErrchk(cudaMallocManaged(&yt, (d)*n * sizeof(coord)));
  coord *VScat;
  gpuErrchk(cudaMallocManaged(&VScat, (d + 1) * n * sizeof(coord)));
  coord *PhiScat;
  gpuErrchk(cudaMallocManaged(&PhiScat, (d + 1) * n * sizeof(coord)));
  int szV = pow(n1 + 2, d) * m;
  coord *VGrid;
  gpuErrchk(cudaMallocManaged(&VGrid, szV * sizeof(coord)));
  coord *PhiGrid;
  gpuErrchk(cudaMallocManaged(&PhiGrid, szV * sizeof(coord)));

  thrust::device_vector<coord> zetaVec(n);

  initKernel<<<Blocks, Threads>>>(VGrid, (coord)0, szV);
  initKernel<<<Blocks, Threads>>>(PhiGrid, (coord)0, szV);

  // Initializing  cufft plans
  cufftHandle plan, plan_rhs;

  int n2 = n1 + 2;
  cufftType type;
  if (sizeof(coord) == 4) {
    type = CUFFT_C2C;

  } else if (sizeof(coord) == 8) {
    type = CUFFT_Z2Z;
  }

  switch (d) {
  case 1: {
    int ng[1] = {(int)n2};
    cufftPlan1d(&plan, n2, type, 1);
    cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n2, NULL, 1, n2, type, d + 1);
    break;
  }
  case 2: {
    int ng[2] = {(int)n2, (int)n2};
    cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, type, 1);
    cufftPlanMany(&plan_rhs, 2, ng, NULL, 1, n2 * n2, NULL, 1, n2 * n2, type,
                  d + 1);
    break;
  }
  case 3: {
    int ng[3] = {(int)n2, (int)n2, (int)n2};
    cufftPlanMany(&plan, 3, ng, NULL, 1, 0, NULL, 1, 0, type, 1);
    cufftPlanMany(&plan_rhs, 3, ng, NULL, 1, n2 * n2 * n2, NULL, 1,
                  n2 * n2 * n2, type, d + 1);
    break;
  }
  }

  ArrayCopy<<<Blocks, Threads>>>(y, yt, n * d);

  ComputeCharges(VScat, y, n, d);

  timer.Stop();

  timeInfo[4] += timer.Elapsed() / 1000.0;

  nuconv(PhiScat, yt, VScat, n, d, m, nGrid, timeInfo, plan, plan_rhs, VGrid,
         PhiGrid);

  timer.Start();
  coord zeta = zetaAndForce(Frep, y, n, d, PhiScat, zetaVec);
  timer.Stop();
  timeInfo[3] = timer.Elapsed() / 1000.0;

  cufftDestroy(plan);
  cufftDestroy(plan_rhs);
  gpuErrchk(cudaFree(PhiGrid));
  gpuErrchk(cudaFree(VGrid));
  gpuErrchk(cudaFree(yt));
  gpuErrchk(cudaFree(VScat));
  gpuErrchk(cudaFree(PhiScat));

  return zeta;
}
