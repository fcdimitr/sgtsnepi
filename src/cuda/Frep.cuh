/*!
  \file   Frep.cuh
  \brief  Implementation for the apprpoximation of the Repulsive term header.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef FREP_CUH
#define FREP_CUH
#include "common.cuh"
#include "nuconv.cuh"

#include "utils_cuda.cuh"


coord computeFrepulsive_gpu(coord *Frep, coord *y, int n, int d, double h,
                            double *timeInfo);
coord computeFrepulsive_GPU(coord *Freph, coord *yh, int n, int d, double h,
                            double *timeInfo);
#endif
