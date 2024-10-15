/*!
  \file   sgtsnepi_mex.cpp
  \brief  MEX entry point of SG-t-SNE-Pi.

  \author Dimitris Floros
  \date   2019-06-22
*/


#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include <sys/time.h>
#include <string>
#include <iostream>

#include "cilk.hpp"

#include "sgtsne.hpp"
#include "../csb/csb_wrapper.hpp"
#include "gradient_descend.hpp"

static const int OPTION_INITIALIZE_CSB   = 0;
static const int OPTION_DESTROY_CSB      = 1;
static const int OPTION_COMPUTE_GRADIENT = 2;
static const int OPTION_COMPUTE_KL       = 3;


void parseInputs( tsneparams &P, double **y, int nrhs, const mxArray *prhs[] ){

  if ( nrhs > 0 && !(mxGetM(prhs[0]) == 0 && mxGetN(prhs[0]) == 0) )
    P.d         = (int) mxGetScalar( prhs[0] );

  if ( nrhs > 1 && !(mxGetM(prhs[1]) == 0 && mxGetN(prhs[1]) == 0) )
    P.lambda    = (int) mxGetScalar( prhs[1] );

  if ( nrhs > 2 && !(mxGetM(prhs[2]) == 0 && mxGetN(prhs[2]) == 0) )
    P.maxIter   = (int) mxGetScalar( prhs[2] );

  if ( nrhs > 3 && !(mxGetM(prhs[3]) == 0 && mxGetN(prhs[3]) == 0) )
    P.earlyIter = (int) mxGetScalar( prhs[3] );

  if ( nrhs > 4 && !(mxGetM(prhs[4]) == 0 && mxGetN(prhs[4]) == 0) )
    P.h         = (double) mxGetScalar( prhs[4] );

  if ( nrhs > 5 && !(mxGetM(prhs[5]) == 0 && mxGetN(prhs[5]) == 0) )
    P.np        = (int) mxGetScalar( prhs[5] );

  if ( nrhs > 6 && !(mxGetM(prhs[6]) == 0 && mxGetN(prhs[6]) == 0) )
    P.alpha     = (double) mxGetScalar( prhs[6] );

  if ( nrhs > 7 && !(mxGetM(prhs[7]) == 0 && mxGetN(prhs[7]) == 0) )
    *y = (double *) mxGetData( prhs[7] );

  if ( nrhs > 8 && !(mxGetM(prhs[8]) == 0 && mxGetN(prhs[8]) == 0) )
    P.dropLeaf = (int) mxGetScalar( prhs[8] );
  
  
}

BiCsb<matval, matidx> *csb = NULL;

void buildCSB( matidx * row, matidx * col, matval * val, 
               int m, int n, int nnz ){
  // build CSB object (with default workers & BETA)
  csb = prepareCSB<matval, matidx>
    ( val, row, col,
      nnz,
      m,
      n,
      0, 0 );
}

void destroyCSB(){
  deallocate(csb);
  csb = NULL;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  double   *dy, *y_in = NULL, *zeta;
  uint32_t nPts, nDim;
  tsneparams params;
  bool flagTime = false;
  double **timeInfo = nullptr;
  size_t nnz;

  // first input is option
  int option = (int) mxGetScalar( prhs[0] );
  
  switch (option){
  case OPTION_INITIALIZE_CSB:
    {
      uint32_t *ir = (uint32_t *) mxGetData( prhs[1] );
      uint32_t *jc = (uint32_t *) mxGetData( prhs[2] );
      double   *vv = (double *)   mxGetData( prhs[3] );
    
      nPts = (uint32_t) mxGetM( prhs[2] ) - 1;
      nnz  = (uint32_t) mxGetM( prhs[1] );
      
      // ---------- prepare local matrices
      matidx *rows = new matidx [nnz];
      matidx *cols = new matidx [nPts+1];
      matval *vals = new matval [nnz];

      std::copy( vv, vv + nnz   , vals );
      std::copy( ir, ir + nnz   , rows );
      std::copy( jc, jc + nPts+1, cols );

      if (csb != NULL) destroyCSB();
      
      buildCSB( rows, cols, vals, nPts, nPts, nnz );

      // delete [] rows;
      // delete [] cols;
      // delete [] vals;
      
    }
    break;

  case OPTION_DESTROY_CSB:
    {
      if (csb != NULL) destroyCSB();
    }
    break;
    
  case OPTION_COMPUTE_GRADIENT:
    {
      if (csb == NULL){
        mexErrMsgTxt("No CSB object");
      }
      y_in = (double *) mxGetData( prhs[1] );
      nDim = (uint32_t) mxGetM( prhs[1] );
      nPts = (uint32_t) mxGetN( prhs[1] );

      double timeFattr = 0.0;
      double timeFrep  = 0.0;

      params.alpha = (double) mxGetScalar( prhs[2] );
      params.d     = nDim;
      params.n     = nPts;

      switch (params.d){
      case 1:
        params.h = 0.5;
        break;
      case 2:
        params.h = 0.7;
        break;
      case 3:
        params.h = 1.2;
        break;
      }

      params.np = getWorkers();

      
      plhs[0] = mxCreateNumericMatrix(params.d, nPts, mxDOUBLE_CLASS, mxREAL);
      plhs[1] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
      dy      = (double *)mxGetData(plhs[0]);
      zeta    = (double *)mxGetData(plhs[1]);

      // mexPrintf("%d %d | %f %f\n", params.n, params.d, params.h, params.alpha);
      zeta[0] = compute_gradient(dy, &timeFrep, &timeFattr, params, y_in, csb);
      
    }
    break;
    
  case OPTION_COMPUTE_KL:
    {
      
    }
    break;
    
  default:
    break;
  }
  
  // ~~~~~~~~~~~~~~~~~~~~ PARSE INPUTS
  // double *vv = (double *)mxGetData(prhs[0]);
  // size_t *ir = mxGetIr( prhs[0] );
  // size_t *jc = mxGetJc( prhs[0] );
  
  // // ---------- get sizes
  // nPts = (uint32_t) mxGetM( prhs[0] );
  // nnz  = mxGetNzmax(prhs[0]);

  // params.n = nPts;

  // // parse inputs
  // parseInputs( params, &y_in, nrhs-1, &prhs[1] );
  
  // // ---------- prepare local matrices
  // matidx *rows = (matidx *) malloc( nnz      * sizeof(matidx) );
  // matidx *cols = (matidx *) malloc( (nPts+1) * sizeof(matidx) );
  // matval *vals = (matval *) malloc( nnz      * sizeof(matval) );

  // std::copy( vv, vv + nnz   , vals );
  // std::copy( ir, ir + nnz   , rows );
  // std::copy( jc, jc + nPts+1, cols );
  
  // // ~~~~~~~~~~~~~~~~~~~~ SETUP OUTPUS
  // plhs[0] = mxCreateNumericMatrix(nPts, params.d, mxDOUBLE_CLASS, mxREAL);

  // if (nlhs > 1) flagTime = true;

  // if (flagTime){
  //   plhs[1] = mxCreateNumericMatrix(6, params.maxIter, mxDOUBLE_CLASS, mxREAL);
  //   timeInfo = (double **) malloc( params.maxIter * sizeof(double *) );
    
  //   double *mxTime = (double *)mxGetData(plhs[1]);

  //   for (int i=0; i<params.maxIter; i++)
  //     timeInfo[i] = &mxTime[i*6];
    
  // }
  
  // // get pointer
  // y_mx = (double *)mxGetData(plhs[0]);

  // // ---------- build sparse_matrix struct
  // sparse_matrix P;
  // P.m = nPts;
  // P.n = nPts;
  // P.nnz = nnz;
  // P.row = rows;
  // P.col = cols;
  // P.val = vals;

  // // ~~~~~~~~~~ setup number of workers
  
  // if (getWorkers() != params.np && params.np > 0)
  //   setWorkers( params.np );

  // params.np = getWorkers();

  
  // // ~~~~~~~~~~~~~~~~~~~~ (OPERATION)
  // double *y = sgtsne( P, params, y_in, timeInfo );

  // // write back to output (transpose for MATLAB)
  // for (int i=0; i<nPts; i++)
  //   for (int j=0; j<params.d; j++)
  //     y_mx[ j*nPts + i ] = y[ i*params.d + j ];
  
  // // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE TEMPORARY MEMORY
  // free( y );
  // if (flagTime) free(timeInfo);
  
}
