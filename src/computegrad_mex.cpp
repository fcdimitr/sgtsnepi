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
#include <cilk/cilk_api.h>

#include "sgtsne.hpp"
#include "../csb/csb_wrapper.hpp"
#include "gradient_descend.hpp"
#include "qq.hpp"

static const int OPTION_INITIALIZE_CSB            = 0;
static const int OPTION_DESTROY_CSB               = 1;
static const int OPTION_COMPUTE_GRADIENT          = 2;
static const int OPTION_COMPUTE_GRADIENT_SEPARATE = 3;
static const int OPTION_COMPUTE_REPULSIVE         = 4;

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
      matidx *rows = (matidx *) malloc( nnz      * sizeof(matidx) );
      matidx *cols = (matidx *) malloc( (nPts+1) * sizeof(matidx) );
      matval *vals = (matval *) malloc( nnz      * sizeof(matval) );

      std::copy( vv, vv + nnz   , vals );
      std::copy( ir, ir + nnz   , rows );
      std::copy( jc, jc + nPts+1, cols );

      if (csb != NULL) destroyCSB();
      
      buildCSB( rows, cols, vals, nPts, nPts, nnz );

      // free(rows);
      // free(cols);
      // free(vals);
      
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
    
  case OPTION_COMPUTE_GRADIENT_SEPARATE:
    {
      if (csb == NULL){
        mexErrMsgTxt("No CSB object");
      }
      y_in = (double *) mxGetData( prhs[1] );
      nDim = (uint32_t) mxGetM( prhs[1] );
      nPts = (uint32_t) mxGetN( prhs[1] );

      double *Fattr, *Frep;

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
      plhs[1] = mxCreateNumericMatrix(params.d, nPts, mxDOUBLE_CLASS, mxREAL);
      plhs[2] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
      
      Frep    = (double *)mxGetData(plhs[0]);
      Fattr   = (double *)mxGetData(plhs[1]);
      zeta    = (double *)mxGetData(plhs[2]);

      csb_pq( NULL, NULL, csb, y_in, Fattr, params.n, params.d, 0, 0, 0 );
      zeta[0] = computeFrepulsive_interp(Frep, y_in, params.n, params.d, params.h, params.np);

    }
    break;

  case OPTION_COMPUTE_REPULSIVE:
    {

      y_in = (double *) mxGetData( prhs[1] );
      nDim = (uint32_t) mxGetM( prhs[1] );
      nPts = (uint32_t) mxGetN( prhs[1] );

      double *Frep;

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

      if (nrhs > 3) params.h = (double) mxGetScalar( prhs[3] );

      params.np = getWorkers();

      
      plhs[0] = mxCreateNumericMatrix(params.d, nPts, mxDOUBLE_CLASS, mxREAL);
      plhs[1] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
      
      Frep    = (double *)mxGetData(plhs[0]);
      zeta    = (double *)mxGetData(plhs[1]);

      zeta[0] = computeFrepulsive_interp(Frep, y_in, params.n, params.d, params.h, params.np);

    }
    break;
    
  default:
    break;
  }
  
}
