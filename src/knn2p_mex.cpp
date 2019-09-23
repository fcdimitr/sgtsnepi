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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  double   *vv;
  size_t   *ir, *jc;
  uint32_t nPts, nn;
  
  // ~~~~~~~~~~~~~~~~~~~~ PARSE INPUTS

  if (nrhs != 3) mexErrMsgTxt("Wrong number of inputs, 3 are required");
  
  // ---------- get data and sizes
  int    * I = (int *)    mxGetData(prhs[0]);
  double * D = (double *) mxGetData(prhs[1]);
  nn         = mxGetM(prhs[0])-1;
  nPts       = mxGetN(prhs[0]);

  // ~~~~~ get perplexity
  double u = mxGetScalar( prhs[2] );

  // ~~~~~~~~~~ run kNN search
  
  sparse_matrix P = perplexityEqualization( I, D, nPts, nn, u );

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
