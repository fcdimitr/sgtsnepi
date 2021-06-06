/*!
  \file   qq.cpp
  \brief  Implementations of repulsive forces computation (QQ component).

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <limits>
#include <cmath>

#include <vector>

#include "timers.hpp"
#include "qq.hpp"
#include "nuconv.hpp"
#include "dataReloc.hpp"

#define N_GRID_SIZE 137

// global vector to report grid sizes
std::vector<int> GLOBAL_GRID_SIZES;

coord computeFrepulsive_exact(coord * frep,
                              coord * pointsX,
                              int N,
                              int d){
  
  coord *zetaVec = new coord [N] ();
  
  cilk_for (int i = 0; i < N; i++) {
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

  cilk::reducer_opadd<coord> zeta_reducer(0.0);
  cilk_for (int i = 0; i < N; i++)
    *zeta_reducer += zetaVec[i];
  coord zeta = zeta_reducer.get_value();

  cilk_for (int i = 0; i < N; i++)
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
  cilk_for (uint32_t i=0; i<nPts; i++){
    for (uint32_t j=0; j<nDim; j++)
      F[iPerm[i]*nDim + j] =
        ( Y[i*nDim+j] * Phi[i*(nDim+1)] - Phi[i*(nDim+1)+j+1] ) / Z;
  }

  return Z;
    
}

int getBestGridSize( int nGrid ){

  // list of FFT sizes that work "fast" with FFTW
  int listGridSize[N_GRID_SIZE] =
    {8,9,10,11,12,13,14,15,16,20,25,26,28,32,33,35,
     36,39,40,42,44,45,48,49,50,52,54,55,56,60,63,64,65,66,70,72,75,
     77,78,80,84,88,90,91,96,98,99,100,104,105,108,110,112,117,120,
     125,126,130,132,135,140,144,147,150,154,156,160,165,168,175,176,
     180,182,189,192,195,196,198,200,208,210,216,220,224,225,231,234,
     240,245,250,252,260,264,270,273,275,280,288,294,297,300,308,312,
     315,320,325,330,336,343,350,351,352,360,364,375,378,385,390,392,
     396,400,416,420,432,440,441,448,450,455,462,468,480,490,495,500,
     504,512};

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

  cilk_for(int i = 0; i < n; i++) {
    for(int j = 0; j < d; j++) {
      y[i*d + j] -= miny[j];
    }
  }

  // ~~~~~~~~~~ find maximum value (across all dimensions) and get grid size
  
  coord maxy = 0;
  for (int i = 0; i < n*d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  
  int nGrid = std::max( (int) std::ceil( maxy / h ), 14 );
  nGrid = getBestGridSize(nGrid);

  GLOBAL_GRID_SIZES.push_back( nGrid );

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

  cilk_for( int i = 0; i < n; i++ ){
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
  
  cilk_for( int i = 0; i < n; i++ ){

    VScat[ i*(d+1) ] = 1.0;
    for ( int j = 0; j < d; j++ )
      VScat[ i*(d+1) + j+1 ] = yt[ i*d + j ];
    
  }

  std::copy( yt, yt + (n*d), yr );

  // ~~~~~~~~~~ run nuConv
  
  if (timeInfo != nullptr)
    nuconv( PhiScat, yt, VScat, ib, cb, n, d, d+1, np, nGrid, &timeInfo[1] );
  else
    nuconv( PhiScat, yt, VScat, ib, cb, n, d, d+1, np, nGrid );

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
