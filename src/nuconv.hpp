/*!
  \file   nuconv.hpp
  \brief  Non-uniform convolution

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef NUCONV_HPP
#define NUCONV_HPP

#include "types.hpp"
#include "utils.hpp"

#define GRID_SIZE_THRESHOLD 20  // Differenet parallelism strategy for small grids

extern "C" {
    void nuconv( coord *PhiScat, coord *y, coord *VScat,
                 uint32_t *ib, uint32_t *cb,
                 int n, int d, int m, int np, int nGridDim,
                 double *timeInfo = nullptr);
}

#endif /* NUCONV_HPP */
