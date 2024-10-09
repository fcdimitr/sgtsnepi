/*!
  \file   csb_wrapper.hpp
  \brief  Wrapper for CSB object and routines.

  \author Dimitris Floros
  \date   2019-07-12
*/


#ifndef _H_EXP_STAT
#define _H_EXP_STAT

#define INDEXTYPE uint32_t


#include <stdint.h>

template <class NT, class IT>
class BiCsb;

INDEXTYPE csb_pq
( double *t_day_csb,double *t_day_csb_tar,
  BiCsb<double, INDEXTYPE> * bicsb,
  double * const x_in,
  double * const y_out,
  int n, int dim, int iter,
  int workers, INDEXTYPE forcelogbeta );

INDEXTYPE csb_pq
( double *t_day_csb,double *t_day_csb_tar,
  BiCsb<float, INDEXTYPE> * bicsb,
  float * const x_in,
  float * const y_out,
  int n, int dim, int iter,
  int workers, INDEXTYPE forcelogbeta );


template <class NT, class IT>
BiCsb<NT,IT> * prepareCSB(NT *vals, IT *rows, IT *cols,
               IT nzmax, IT m, IT n,
               int workers, int forcelogbeta);


float tsne_cost
( BiCsb<float, INDEXTYPE> * bicsb,
  float * const x_in, int N,
  int dim, float alpha, float zeta);


double tsne_cost
( BiCsb<double, INDEXTYPE> * bicsb,
  double * const x_in, int N,
  int dim, double alpha, double zeta);

template <class NT, class IT>
void deallocate( BiCsb<NT,IT> * bicsb );

#endif


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
