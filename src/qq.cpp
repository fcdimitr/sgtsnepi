/*!
  \file   qq.cpp
  \brief  Implementations of repulsive forces computation (QQ component).

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <limits>
#include <cmath>

#include <vector>

#include "cilk.hpp" 
#include "timers.hpp"
#include "qq.hpp"
#include "nuconv.hpp"
#include "dataReloc.hpp"
#include "opadd_reducer.hpp"

int N_GRID_SIZE = 26;
// #define N_GRID_SIZE 57
// #define N_GRID_SIZE 7

// global vector to report grid sizes
std::vector<double> GLOBAL_GRID_SIZES;


int *listGridSize;
// {16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108,
//  128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432,
//  486, 512};

coord computeFrepulsive_exact(coord * frep,
                              coord * pointsX,
                              int N,
                              int d){
  
  coord *zetaVec = new coord [N] ();
  
  CILK_FOR (int i = 0; i < N; i++) {
    coord Yi[10] = {0};
    for (int dd = 0; dd < d; dd++ )
      Yi[dd] = pointsX[i*d + dd];

    coord Yj[10] = {0};
    
    for(int j = 0; j < N; j++) {
      
      if(i != j) {

        coord dist = 0.0;
        for (int dd = 0; dd < d; dd++ ){
           Yj[dd] = pointsX[j*d + dd];
           dist += (Yj[dd] - Yi[dd]) * (Yj[dd] - Yi[dd]);
        }

        for (int dd = 0; dd < d; dd++ ){
	  frep[i*d + dd] += (Yi[dd] - Yj[dd]) /
	       ( (1 + dist)*(1 + dist) );
	}
        
        zetaVec[i] += 1.0 / (1.0 + dist);
        
      }
    }
  }

  opadd_reducer<coord> zeta_reducer = 0.0;
  CILK_FOR (int i = 0; i < N; i++)
    zeta_reducer += zetaVec[i];

  coord zeta = static_cast<coord>(zeta_reducer);

  CILK_FOR (int i = 0; i < N; i++)
    for (int j = 0; j < d; j++)
      frep[(i*d) + j] /= zeta;

  delete [] zetaVec;
  return zeta;
  
}

//! Compute normalization term and repulsive forces.
/*!
*/
template<typename dataval>
dataval zetaAndForce( dataval * const F,            // Forces
                      const dataval * const Y,      // Coordinates
                      const dataval * const Phi,    // Values
                      const uint32_t * const iPerm,// Permutation 
                      const uint32_t nPts,         // #points
                      const uint32_t nDim ) {      // #dimensions

  dataval Z = 0;
  
  // compute normalization term
  for (uint32_t i=0; i<nPts; i++){
    dataval Ysq = 0;
    for (uint32_t j=0; j<nDim; j++){
      Ysq += Y[i*nDim+j] * Y[i*nDim+j];
      Z -= 2 * ( Y[i*nDim+j] * Phi[i*(nDim+1)+j+1] );
    }
    Z += ( 1 + 2*Ysq ) * Phi[i*(nDim+1)];
  }

  Z = Z-nPts;
  
  // Compute repulsive forces
  CILK_FOR (uint32_t i=0; i<nPts; i++){
    for (uint32_t j=0; j<nDim; j++)
      F[iPerm[i]*nDim + j] =
        ( Y[i*nDim+j] * Phi[i*(nDim+1)] - Phi[i*(nDim+1)+j+1] ) / Z;
  }

  return Z;
    
}

/**
 * @brief      Returns grid size that is efficient with FFTW
 *
 * @details    The function returns the grid size minus 2; the grid is padded in the nuconv.cpp routines
 *
 */
int getBestGridSize( int nGrid ){

  // list of FFT sizes that work "fast" with FFTW
  // int listGridSize[N_GRID_SIZE] =
  //   {16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50,
  //    54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120,
  //    125, 128, 135, 144, 150, 160, 162, 180, 192, 200,
  //    216, 225, 240, 243, 250, 256, 270, 288, 300, 320,
  //    324, 360, 375, 384, 400, 405, 432, 450, 480, 486,
  //    500, 512};

  // int listGridSize[N_GRID_SIZE] =
  //   {16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108,
  //    128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432,
  //    486, 512};

  // int listGridSize[N_GRID_SIZE] =
  //   {8,16,32,64,128,256,512};

  // select closest (larger) size for given grid size
  for (int i=0; i<N_GRID_SIZE; i++)
    if ( (nGrid+2) <= listGridSize[i] )
      return listGridSize[i]-2;

  return listGridSize[N_GRID_SIZE-1]-2;
  
}


coord computeFrepulsive_interp(coord * Frep,
                               coord * y,
                               int n,
                               int d,
                               double h,
                               int np,
                               int single,
                               double *timeInfo){

  // ~~~~~~~~~~ make temporary data copies
  coord *yt = new coord [n*d];
  coord *yr = new coord [n*d];

  struct timeval start;
  
  // ~~~~~~~~~~ move data to (0,0,...)
  coord miny[d];
  for (int j = 0; j < d; j++)
    miny[j] = std::numeric_limits<coord>::infinity();
  
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++)
      miny[j] = miny[j] > y[i*d + j] ? y[i*d + j] : miny[j];

  CILK_FOR(int i = 0; i < n; i++) {
    for(int j = 0; j < d; j++) {
      y[i*d + j] -= miny[j];
    }
  }

  // ~~~~~~~~~~ find maximum value (across all dimensions) and get grid size
  
  coord maxy = 0;
  for (int i = 0; i < n*d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;

  int nGrid;

  if (h>0){
    nGrid = std::max( (int) std::ceil( maxy / h ), 14 );
    nGrid = getBestGridSize(nGrid);
  } else {
    nGrid = (int) -h;
  }

  GLOBAL_GRID_SIZES.push_back( (double) (nGrid+2) );
  GLOBAL_GRID_SIZES.push_back( maxy );

#ifdef VERBOSE
  std::cout << "Grid: " << nGrid << " h: " << h << std::endl;
#endif

  // ~~~~~~~~~~ setup inputs to nuConv
  
  std::copy( y, y + (n*d), yt );

  coord *VScat    = new coord [n*(d+1)];
  coord *PhiScat  = new coord [n*(d+1)] ();
  uint32_t *iPerm = new uint32_t [n];
  uint32_t *ib    = new uint32_t [nGrid] ();
  uint32_t *cb    = new uint32_t [nGrid] ();

  CILK_FOR( int i = 0; i < n; i++ ){
    iPerm[i] = i;
  }

  start = tsne_start_timer();
  relocateCoarseGrid( &yt,        
                      &iPerm,
                      ib,
                      cb,
                      n, 
                      nGrid,
                      d,
                      np );
  if (timeInfo != nullptr)
    timeInfo[0] = tsne_stop_timer("Gridding", start);
  else
    tsne_stop_timer("Gridding", start);

  // ----- setup VScat (value on scattered points)
  
  CILK_FOR( int i = 0; i < n; i++ ){

    VScat[ i*(d+1) ] = 1.0;
    for ( int j = 0; j < d; j++ )
      VScat[ i*(d+1) + j+1 ] = yt[ i*d + j ];
    
  }

  std::copy( yt, yt + (n*d), yr );

  // ~~~~~~~~~~ run nuConv
  
  if (timeInfo != nullptr)
    nuconv( PhiScat, yt, VScat, ib, cb, n, d, d+1, np, nGrid, single, &timeInfo[1] );
  else
    nuconv( PhiScat, yt, VScat, ib, cb, n, d, d+1, np, nGrid, single );

  // ~~~~~~~~~~ compute Z and repulsive forces
  
  start = tsne_start_timer();
  coord zeta = zetaAndForce( Frep, yr, PhiScat, iPerm, n, d );
  if (timeInfo != NULL)
    timeInfo[4] = tsne_stop_timer("F&Z", start);
  else
    tsne_stop_timer("F&Z", start);
  
  delete [] yt;
  delete [] yr;
  delete [] VScat;
  delete [] PhiScat;
  delete [] iPerm;
  delete [] ib;
  delete [] cb;
  return zeta;
  
}
