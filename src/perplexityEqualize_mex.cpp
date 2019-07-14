/*!
  \file   sgtsnepi_mex.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-22
*/


#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include <sys/time.h>
#include <string>
#include <iostream>
#include <cilk/cilk_api.h>
#include <cmath>
#include <flann/flann.h>

#include "sgtsne.hpp"

//! Compute the approximate all-kNN graph of the input data points
/*!  
  Compute the k-nearest neighbor dataset points of every point in
  the datsaet set using approximate k-nearest neighbor search (FLANN).

*/
void allKNNsearch(int * IDX,        //!< [k-by-N] array with the neighbor IDs
                  double * DIST,    //!< [k-by-N] array with the neighbor distances
                  double * dataset, //!< [L-by-N] array with coordinates of data points
                  int N,            //!< [scalar] Number of data points N
                  int dims,         //!< [scalar] Number of dimensions L
                  int kappa) {      //!< [scalar] Number of neighbors k

  
  struct FLANNParameters p; 

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 8;
  // p.log_level = FLANN_LOG_INFO;
  p.checks = 100;
  
  // -------- Run a kNN search
  flann_find_nearest_neighbors_double(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  double   *X, *vv;
  size_t   *ir, *jc;
  uint32_t nPts, nDim;
  size_t nnz;
  
  // ~~~~~~~~~~~~~~~~~~~~ PARSE INPUTS

  if (nrhs != 2) mexErrMsgIdAndTxt("1", "Wrong number of inputs, 2 are required");
  
  // ---------- get sizes
  X = (double *) mxGetData( prhs[0] );
  nDim = (uint32_t) mxGetM( prhs[0] );
  nPts = (uint32_t) mxGetN( prhs[0] );

  // ~~~~~ get perplexity
  double u = mxGetScalar( prhs[1] );

  int nn = std::ceil( 3*u );
  
  // ~~~~~~~~~~ run kNN search

  std::cout << "Running k-neareast neighbor search for " << nn << " neighbors..."
            << std::flush;
  
  double * D = (double *)malloc(nPts * (nn + 1) * sizeof(double));
  int    * I = (int *)malloc(nPts * (nn + 1) * sizeof(int));

  allKNNsearch(I, D, X, nPts, nDim, nn+1);

  std::cout << "DONE" << std::endl;

  sparse_matrix P = perplexityEqualization( I, D, nPts, nn, u );

  free( D ); free( I );
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP OUTPUS
  plhs[0] = mxCreateSparse(nPts,nPts,P.nnz,mxREAL);

  vv = mxGetPr(plhs[0]);
  ir = mxGetIr(plhs[0]);
  jc = mxGetJc(plhs[0]);
  
  // ---------- copy data to MATLAB
  std::copy( P.val, P.val + P.nnz , vv );
  std::copy( P.row, P.row + P.nnz , ir );
  std::copy( P.col, P.col + nPts+1, jc );
  

  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE TEMPORARY MEMORY
  free_sparse_matrix(&P);
  
}
