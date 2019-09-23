/*!
  \file   flannsearch_mex.cpp
  \brief  

  Approximate kNN search using FLANN.

  \author Dimitris Floros
  \date   2019-09-20
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
  p.trees = 16;
  p.checks = 300;
  p.random_seed = 0;
  
  // -------- Run a kNN search
  flann_find_nearest_neighbors_double(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  double   *X;
  uint32_t nPts, nDim;
  
  // ~~~~~~~~~~~~~~~~~~~~ PARSE INPUTS

  if (nrhs != 2) mexErrMsgIdAndTxt("1", "Wrong number of inputs, 2 are required");
  
  // ---------- get sizes
  X = (double *) mxGetData( prhs[0] );
  nDim = (uint32_t) mxGetM( prhs[0] );
  nPts = (uint32_t) mxGetN( prhs[0] );

  // ~~~~~ get perplexity
  int nn = (int) mxGetScalar( prhs[1] );

  // ~~~~~~~~~~ run kNN search

  std::cout << "Running k-neareast neighbor search for " << nn << " neighbors..."
            << std::flush;

  plhs[0] = mxCreateNumericMatrix((nn+1), nPts, mxINT32_CLASS, mxREAL);
  plhs[1] = mxCreateNumericMatrix((nn+1), nPts, mxDOUBLE_CLASS, mxREAL);

  int    * I = (int *)    mxGetData(plhs[0]);
  double * D = (double *) mxGetData(plhs[1]);

  allKNNsearch(I, D, X, nPts, nDim, nn+1);

  std::cout << "DONE" << std::endl;
  
}
