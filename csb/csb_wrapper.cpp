/*!
  \file   csb_wrapper.cpp
  \brief  Wrapper for CSB object and routines.

  \author Dimitris Floros
  \date   2019-07-12
*/

// --------------------------------------------------
// Include headers

#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include <cstring>

#include "csb_wrapper.hpp"

#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"

#define RHS3DIM 3


/*
 * Although this is a template, the result is always a CSB object.
 * This is a hack to get arount the CSB library.
 */
template <class NT, class IT>
BiCsb<NT,IT> * prepareCSB( NT *values, IT *rows, IT *cols,
                IT nzmax, IT m, IT n,
                int workers, int forcelogbeta ){

  // generate CSC object (CSB definitions)
  Csc<NT, IT> * csc;
  csc = new Csc<NT, IT>();

  csc->SetPointers( cols, rows, values, nzmax, m, n, 0 );
   
  if (workers == 0)
    workers = __cilkrts_get_nworkers();
  else{
    std::string sworkers = std::to_string(workers);
    __cilkrts_set_param("nworkers", sworkers.c_str());
  }
  
  BiCsb<NT,IT> *bicsb = new BiCsb<NT, IT>(*csc, workers, forcelogbeta);

  // clean CSB-type CSC object
  delete( csc );

  return bicsb;
}

template <class NT, class IT>
void deallocate( BiCsb<NT,IT> * bicsb ){

  // generate CSC object (CSB definitions)
  delete bicsb;
}

static double getMillisecondsExp( struct timeval begin, struct timeval end ) {

  return
    ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
    ((double) (end.tv_usec - begin.tv_usec) / 1000 );
}

// --------------------------------------------------
// Function definitions

INDEXTYPE csb_pq
( double *t_day_csb,double *t_day_csb_tar,
  BiCsb<double, INDEXTYPE> * bicsb,
  double * const x_in,
  double * const y_out,
  int n, int dim, int iter,
  int workers, INDEXTYPE forcelogbeta ) {
  
  struct timeval begin, end;
  
  // find CSB block size
  INDEXTYPE actual_beta = bicsb->getBeta();

  // prepare template type for CSB routine
  typedef PTSR<double,double> PTDD;

  /////////////////////
  // Run experiments //
  /////////////////////
  switch (dim){

  case 1:
    bicsb_tsne1D<PTDD>(*bicsb, x_in, y_out);
    break;

  case 2:
    bicsb_tsne2D<PTDD>(*bicsb, x_in, y_out);
    break;
    
  case 3:
    bicsb_tsne<PTDD>(*bicsb, x_in, y_out);
    break;

  case 4:
    bicsb_tsne4D<PTDD>(*bicsb, x_in, y_out);
    break;
    
  }
    
  // return actual beta
  return actual_beta;
  
}

INDEXTYPE csb_pq
( double *t_day_csb,double *t_day_csb_tar,
  BiCsb<float, INDEXTYPE> * bicsb,
  float * const x_in,
  float * const y_out,
  int n, int dim, int iter,
  int workers, INDEXTYPE forcelogbeta ) {
  
  struct timeval begin, end;
  
  // find CSB block size
  INDEXTYPE actual_beta = bicsb->getBeta();

  // prepare template type for CSB routine
  typedef PTSR<float,float> PTDD;

  /////////////////////
  // Run experiments //
  /////////////////////
  switch (dim){

  case 1:
    bicsb_tsne1D<PTDD>(*bicsb, x_in, y_out);
    break;

  case 2:
    bicsb_tsne2D<PTDD>(*bicsb, x_in, y_out);
    break;
    
  case 3:
    bicsb_tsne<PTDD>(*bicsb, x_in, y_out);
    break;

  case 4:
    bicsb_tsne4D<PTDD>(*bicsb, x_in, y_out);
    break;

  }
    
  // return actual beta
  return actual_beta;
  
}

float tsne_cost
( BiCsb<float, INDEXTYPE> * bicsb,
  float * const x_in, int N,
  int dim, float alpha, float zeta) {
  
  float *y_out = (float*) calloc(N, sizeof(float));
  
  // prepare template type for CSB routine
  typedef PTSR<float,float> PTDD;

  /////////////////////
  // Run SPMV        //
  /////////////////////

  bicsb_tsne_cost<PTDD>(*bicsb, x_in, y_out, dim, alpha, zeta);
  float total_cost = 0;
  
  for (int i =0; i<N; i++) total_cost += y_out[i];
  // NT total_cost = __sec_reduce_add( y_out[0:N] );
  free( y_out );

  return total_cost;
  
}

double tsne_cost
( BiCsb<double, INDEXTYPE> * bicsb,
  double * const x_in, int N,
  int dim, double alpha, double zeta) {
  
  double *y_out = (double*) calloc(N, sizeof(double));
  
  // prepare template type for CSB routine
  typedef PTSR<double,double> PTDD;

  /////////////////////
  // Run SPMV        //
  /////////////////////

  bicsb_tsne_cost<PTDD>(*bicsb, x_in, y_out, dim, alpha, zeta);
  double total_cost = 0;
  
  for (int i =0; i<N; i++) total_cost += y_out[i];
  // NT total_cost = __sec_reduce_add( y_out[0:N] );
  free( y_out );

  return total_cost;
  
}


// ***** EXPLICIT INSTATIATION

template
BiCsb<float, uint32_t> * prepareCSB
(float *vals, uint32_t *rows, uint32_t *cols,
 uint32_t nzmax, uint32_t m, uint32_t n,
 int workers, int forcelogbeta );

template
BiCsb<double, uint32_t> * prepareCSB
(double *vals, uint32_t *rows, uint32_t *cols,
 uint32_t nzmax, uint32_t m, uint32_t n,
 int workers, int forcelogbeta );

template
void deallocate( BiCsb<double,uint32_t> * bicsb );

template
void deallocate( BiCsb<float,uint32_t> * bicsb );


/**------------------------------------------------------------
*
* AUTHORS
*
*   Dimitris Floros                         fcdimitr@auth.gr
*
* VERSION
* 
*   1.0 - July 13, 2018
*
* CHANGELOG
*
*   1.0 (Jul 13, 2018) - Dimitris
*       * all interaction types in one file
*       
* ----------------------------------------------------------*/
