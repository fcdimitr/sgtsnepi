/**
 * @file   utils.hpp
 * @author Dimitris Floros <fcdimitr@auth.gr>
 * @date   Wed Sep 20 13:51:08 2017
 * 
 * @brief  Utility functions
 *
 * Various independed functions used throught the project
 * 
 * 
 */

#ifndef _H_UTILS
#define _H_UTILS
#include "cs.hpp"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <limits.h>
#include <assert.h>
#include <algorithm>    // std::replace

/**
 * @brief Verifies equality between vectors
 * 
 * Verifies that input vectors are equal (with respect to ERR_THRES)
 * 
 * @param f_new The new vector that need to be checked
 * @param f_gold The gold, correct vector
 * @param n The number of elements
 * @param dim Dimension of each element
 * @param ERR_THRES Error threshold
 */
template <typename T>
void verifyVectorEqual(T const * const f_new, T const * const f_gold,
                       CS_INT n, CS_INT dim, double const ERR_THRES);

/** 
 * @brief Prints dense matrix
 * 
 * @param matrix The matrix to print
 * @param m Number of rows
 * @param n Number of columns
 */
void printDenseMatrix( double *matrix, CS_INT m, CS_INT n );

/** 
 * @brief Transforms dense banded to CS sparse
 * 
 * @param B Input dense
 * @param n Size of matrix
 * @param b Bandwidth (nnz per row)
 * 
 * @return 
 */
cs *band2sparse( double *B, CS_INT n, CS_INT b );

/** 
 * @brief Transforms dense banded to CS sparse with limited nnz
 * elements
 * 
 * @param B Input dense
 * @param n Size of matrix
 * @param b Bandwidth (nnz per row)
 * @param lim Total nnz limitation
 * 
 * @return 
 */
cs *band2sparseLim( double *B, CS_INT n, CS_INT b, CS_INT lim );

/** 
 * @brief Generates a banded matrix
 * 
 * @param n Size of matrix
 * @param b Bandwidth
 * 
 * @return Banded matrix
 */
double *generateBanded( CS_INT n, CS_INT b );

cs *genSymBandSparse( CS_INT n, CS_INT b, CS_INT lim );

void printMinTime( double *x, CS_INT n );

void exportTime2csv( double *x, FILE *fp, CS_INT n );

cs *make_sym (cs *A);

template<typename T>
T *permuteDataPoints( T* x, CS_INT *p, CS_INT n, CS_INT ldim );

static double tic (void) {
  return (clock () / (double) CLOCKS_PER_SEC);
}

static double toc (double t) {
  double s = tic () ; return (CS_MAX (0, s-t));
}

static double getMilliseconds( struct timeval begin, struct timeval end ) {

  return
    ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
    ((double) (end.tv_usec - begin.tv_usec) / 1000 );
}

// [DEPRECATED]
// void setThreadsNum(int nworkers);

void extractDimensions(double *y, double *x, CS_INT N, CS_INT ldim, CS_INT d);

/** 
 * Extract upper triangular from a sparse matrix in SuiteSparse CSC.
 * 
 * @param A Original sparse matrix
 * 
 * @return New sparse matrix in CSC with only upper triangular
 */
cs * triu( cs const * const A );

/**
 * Check whether string a starts with string b
 */
CS_INT StartsWith(const char *a, const char *b);

#endif
