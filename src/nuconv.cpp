/*!
  \file   nuconv.cpp
  \brief  Non-uniform convolution

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <cilk/cilk.h>
#include <cilk/reducer_max.h>
#include <limits>
#include <cmath>
#include <vector>

#include "nuconv.hpp"
#include "timers.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "gridding.cpp"
#include "non_periodic_conv.cpp"

void nuconv( coord *PhiScat, coord *y, coord *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim,
             int single,
             double *timeInfo){

  struct timeval start;
  
  // ~~~~~~~~~~ normalize coordinates and scale to [0,ng-1] (inside bins)

  cilk::reducer< cilk::op_max<coord> > maxy_reducer;
  cilk_for (int i = 0; i < n*d; i++)
    maxy_reducer->calc_max( y[i] );
  coord maxy = maxy_reducer.get_value();

  cilk_for (int i = 0; i < n*d; i++) {
    y[i] /= maxy;
    y[i] -= (y[i] == 1) * std::numeric_limits<coord>::epsilon();
    y[i] *= (nGridDim-1);
  }

  for (int i = 0; i< n*d; i++)
    if ( (y[i] >= nGridDim-1) || (y[i] < 0) ) exit(1);

  // ~~~~~~~~~~ find exact h
  
  double h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon() );

  
  // ~~~~~~~~~~ scat2grid
  int szV = pow( nGridDim+2, d ) * m;

  ::std::vector<coord> VGrid (szV*np, 0);

  start = tsne_start_timer();
  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g1d( VGrid.data(), y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g1drb( VGrid.data(), y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;
    
  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g2d( VGrid.data(), y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g2drb( VGrid.data(), y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g3d( VGrid.data(), y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g3drb( VGrid.data(), y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;
    
  }

  // ----- reduction across processors
  cilk_for( int i=0; i < szV ; i++ )
    for (int j=1; j<np; j++)
      VGrid[i] += VGrid[ j*szV + i ];

  VGrid.resize( szV );

  if (timeInfo != nullptr)
    timeInfo[0] = tsne_stop_timer("S2G", start);
  else
    tsne_stop_timer("S2G", start);

  
  // ~~~~~~~~~~ grid2grid
  coord *PhiGrid = new coord [szV] ();
  uint32_t * const nGridDims = new uint32_t [d] ();
  for (int i = 0; i < d; i++)
    nGridDims[i] = nGridDim + 2;

  start = tsne_start_timer();

  if (single)
    switch (d) {

      case 1:
        conv1dnopad_f( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
        break;

      case 2:
        conv2dnopad_f( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
        break;

      case 3:
        conv3dnopad_f( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
        break;

    }
  else
    switch (d) {

      case 1:
        conv1dnopad( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
        break;

      case 2:
        conv2dnopad( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
        break;

      case 3:
        conv3dnopad( PhiGrid, VGrid.data(), h, nGridDims, m, d, np );
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
  // VGrid freed by ::std::vector destructor
  delete [] PhiGrid;
  delete [] nGridDims;
  
}
