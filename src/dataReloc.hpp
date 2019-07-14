/*!
  \file   dataReloc.hpp
  \brief  Fast data relocation modules.

  \author Dimitris Floros
  \date   2019-06-24
*/


#ifndef DATARELOC_HPP
#define DATARELOC_HPP

#include "types.hpp"

//! Coarse-grid quantization and data relocation
/*!
*/
template<typename dataval>
void relocateCoarseGrid( dataval  **  Y, // Scattered point coordinates
                         uint32_t **  iPerm, // Data relocation permutation
                         uint32_t *ib, // Starting index of box (along last dimension)
                         uint32_t *cb, // Number of scattered points per box (along last dimension)
                         const uint32_t nPts, // Number of data points
                         const uint32_t nGridDim, // Grid dimensions (+1)
                         const uint32_t nDim, // Number of dimensions
                         const uint32_t np ); // Number of processors

#endif /* DATARELOC_HPP */
