/**
 * @file   csr_routines.cpp
 * @author fcdimitr <fcdimitr@auth.gr>
 * @date   Thu Jul 19, 2018
 * 
 * @brief Implementations for stationary and nonstationary
 * computations using CSR storage format.
 *
 * @version 1.0
 *
 * 
 */

#include "csr_routines.hpp"
#include "cilk/cilk.h"
#include "mkl.h"

#ifdef PARFLG
#define FOR cilk_for
#else
#define FOR for
#endif


void computeSubDistSparse( double       *       Fattr,
                           double       * const Y,
                           double const * const p_sp,
                           int          *       ir,
                           int          *       jc,
                           int    const         n,   
                           int    const         d) {

  // loop over rows of matrix (cilk_for or for)
  FOR (int i = 0; i < n; i++) {
      
    // loop over rows of matrix (cilk_for or for)
    // FOR (int i = 0; i < n; i++) {
    
    double Fi[3] = {0};
    double Yi[3];

    const int nnzi = jc[i+1] - jc[i];       
      
    Yi[:] = Y[ (i*d) + 0:d ];
    
    // for each non zero element of row i
    for (int k = 0; k < nnzi; k++) {

      double Ydij[3];
      const int idx = jc[i]+k;
      const int j   = (ir[idx]);
      
      // compute on-the-fly vector Yi - Yj
      Ydij[:] = Yi[:] - Y[ (j*d) + 0:d ];

      // compute euclidean distance between Yi and Yj
      double dist = __sec_reduce_add( Ydij[:]*Ydij[:] );

      // P(i,j) / ( 1 + dist(Y[i,:],Y[j,:]) ) 
      const double p_times_q = p_sp[idx] / (1+dist);

      // Fi += P[i,j] * Q[i,j] * (Y[i,:] - Y[j,:])
      Fi[:] += p_times_q * ( Ydij[:] );
      
    }

    // update final output vector F[i,:]
    Fattr[ (i*d) + 0:d ] = Fi[:];
    
  }

}
    


// --------------------------------------------------
// ---------- ?CSRMV: MATRIX VECTOR PRODUCT USING SPARSE CSR

void sparseComputationCSR( double              * const y,
                           double        const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           double        const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec){

  char transa = 'n';

  MKL_INT m = n;

  for ( int jj = 0; jj < nOfVec; jj++)

    mkl_cspblas_dcsrgemv (&transa , &m , values, rows, columns , &(x[jj*n]), &(y[jj*n]) );
  
}

void sparseComputationCSR( float               * const y,
                           float         const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           float         const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec){

  char transa = 'n';

  MKL_INT m = n;

  for ( int jj = 0; jj < nOfVec; jj++)

    mkl_cspblas_scsrgemv (&transa , &m , values , rows , columns , &(x[jj*n]), &(y[jj*n]) );
  
}


// ==================================================
// === MATRIX-MATRIX PRODUCT ROUTINES

// --------------------------------------------------
// --- CSR MATRIX-MATRIX PRODUCT

 void sparseMatrixMatrixComputationCSR( double              * const y,
                                        double        const * const values,
                                        int           const * const columns,
                                        int           const * const rows,
                                        double        const * const x,
                                        unsigned int          const n,
                                        unsigned int          const nOfVec){

   char transa[1] = {'N'};

   char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};

   MKL_INT m     = (MKL_INT) n;
   MKL_INT nVecs = (MKL_INT) nOfVec;
   MKL_INT k     = (MKL_INT) n;
  
   double alpha = 1;
   double beta  = 0;

   mkl_dcsrmm (transa, &m, &nVecs, &k, &alpha, matdescra, values, columns,
               rows, &(rows[1]), x, &nVecs, &beta, y, &nVecs );

 }

 void sparseMatrixMatrixComputationCSR( float               * const y,
                                        float         const * const values,
                                        int           const * const columns,
                                        int           const * const rows,
                                        float         const * const x,
                                        unsigned int          const n,
                                        unsigned int          const nOfVec){

   char transa[1] = {'N'};

   char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};

   MKL_INT m     = (MKL_INT) n;
   MKL_INT nVecs = (MKL_INT) nOfVec;
   MKL_INT k     = (MKL_INT) n;
  
   float alpha = 1;
   float beta  = 0;
    
   mkl_scsrmm (transa, &m, &nVecs, &k, &alpha, matdescra, values, columns,
               rows, &(rows[1]), x, &nVecs, &beta, y, &nVecs );

 }


 // ==================================================
 // CUSTOM IMPLEMENTATION OF CSR

void computeSubDistSparse_spmv( double       *       Fattr,
                                double       * const Y,
                                double const * const p_sp,
                                int          *       ir,
                                int          *       jc,
                                int    const         n,   
                                int    const         d) {
  
  // loop over rows of matrix (cilk_for or for)
  FOR (int i = 0; i < n; i++) {

    double Fi[1] = {0};

    const int nnzi = jc[i+1] - jc[i];

    // for each nnz of row i
    for (int k = 0; k < nnzi; k++) {

      const int idx = jc[i] + k;
      const int j   = ir[idx];

      // Fi += P[i,j] * Y[j,:]
      Fi[:] += p_sp[idx] * Y[ (j*d) + 0:d ];
      
    }

    // updated final ouptut vector F[i,:]
    Fattr[ (i*d) + 0:d ] = Fi[:];
      
  }

}




/* **********************************************************************
 *
 * AUTHORS
 *
 *   Dimitris Floros                         fcdimitr@auth.gr
 *
 * VERSION
 *
 *   1.1 - October 25, 2017
 *
 * CHANGELOG
 *
 *   1.1 (Oct 25, 2017) - Dimitris
 *      * added multiple different parallelism options
 *        - different grainsizes
 *        - openmp
 *   1.0 (Oct 18, 2017) - Dimitris
 *      * fixed rows --> columns notation ( i <--> j )
 *      * cleaned up and simplified code
 *   0.1 (???) - ???
 *      * initial implementation
 *
 * ********************************************************************** */
