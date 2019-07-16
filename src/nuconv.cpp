/*!
  \file   nuconv.cpp
  \brief  Non-uniform convolution

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <cilk/cilk.h>
#include <limits>
#include <cmath>

#include "nuconv.hpp"
#include "timers.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "gridding.cpp"
#include "non_periodic_conv.cpp"

void nuconv( coord *PhiScat, coord *y, coord *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim,
             double *timeInfo){

  struct timeval start;
  
  // ~~~~~~~~~~ normalize coordinates (inside bins)
  coord maxy = 0;
  for (int i = 0; i < n*d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  
  y[0:n*d] /= maxy;

  // ~~~~~~~~~~ scale them from 0 to ng-1
  
  if (1 == y[0:n*d])
    y[0:n*d] = y[0:n*d] - std::numeric_limits<coord>::epsilon();
  
  y[0:n*d] *= (nGridDim-1);

  for (int i = 0; i< n*d; i++)
    if ( (y[i] >= nGridDim-1) || (y[i] < 0) ) exit(1);

  // ~~~~~~~~~~ find exact h
  
  double h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon() );

  
  // ~~~~~~~~~~ scat2grid
  int szV = pow( nGridDim+2, d ) * m;
  coord *VGrid = static_cast<coord *> ( calloc( szV * np, sizeof(coord) ) );

  start = tsne_start_timer();
  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g1d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g1drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;
    
  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g2d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g2drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g3d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g3drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;
    
  }

  // ----- reduction across processors
  cilk_for( int i=0; i < szV ; i++ )
    for (int j=1; j<np; j++)
      VGrid[i] += VGrid[ j*szV + i ];

  VGrid = static_cast<coord *> ( realloc( VGrid, szV*sizeof(coord) ) );  

  if (timeInfo != nullptr)
    timeInfo[0] = tsne_stop_timer("S2G", start);
  else
    tsne_stop_timer("S2G", start);

  
  // ~~~~~~~~~~ grid2grid
  coord *PhiGrid = static_cast<coord *> ( calloc( szV, sizeof(coord) ) );
  uint32_t * const nGridDims = new uint32_t [d]();
  nGridDims[0:d] = nGridDim + 2;

  start = tsne_start_timer();
  
  switch (d) {

  case 1:
    conv1dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  case 2:
    conv2dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  case 3:
    conv3dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  }

  if (timeInfo != nullptr)
    timeInfo[1] = tsne_stop_timer("G2G", start);
  else
    tsne_stop_timer("G2G", start);

  
  // ~~~~~~~~~~ grid2scat
  start = tsne_start_timer();
  
  switch (d) {

  case 1:
    g2s1d( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;
    
  case 2:
    g2s2d( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;

  case 3:
    g2s3d( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;
    
  }

  if (timeInfo != nullptr)
    timeInfo[2] = tsne_stop_timer("G2S", start);
  else
    tsne_stop_timer("G2S", start);

  // ~~~~~~~~~~ deallocate memory
  free( VGrid );
  free( PhiGrid );

  delete nGridDims;
  
}
