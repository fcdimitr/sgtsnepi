/* **********************************************************************
 *
 * CSR_ROUTINES
 * ----------------------------------------------------------------------
 *
 *   Header file containing definition of CSR routines for experiments.
 *
 * ********************************************************************** */

#ifndef _CSR_ROUTINES
#define _CSR_ROUTINES


/**
   Single precision sparse matrix vector product using MKL SCSCMV,
   with matrix stored in Compressed Sparse Column format (CSC).

   @param y       Output vector y (result of matrix-vector product)
   @param values  Vector containing the nnz elements of sparse matrix
   @param rows    Rows vector
   @param columns Columns vector
   @param x       Input vector x
   @param b       Bandwidth size (nnz for each row)
 */
void sparseComputationCSR( double              * const y,
                           double        const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           double        const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec);

void sparseComputationCSR( float               * const y,
                           float         const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           float         const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec);


// --------------------------------------------------
// GENERAL SPARSE MATRIX VECTOR PRODUCT -- CSR FORMAT (MKL_?CSRMV)

/**
   Double precision sparse matrix vector product using MKL DCSRMV,
   with matrix stored in Compressed Sparse Row format (CSR).

   @param y       Output vector y (result of matrix-vector product)
   @param values  Vector containing the nnz elements of sparse matrix
   @param columns Columns vector
   @param rows    Rows vector
   @param x       Input vector x
   @param b       Bandwidth size (nnz for each row)
   @param nOfVec  Number of vectors
 */
void sparseMatrixMatrixComputationCSR( double              * const y,
                                       double        const * const values,
                                       int           const * const columns,
                                       int           const * const rows,
                                       double        const * const x,
                                       unsigned int          const n,
                                       unsigned int          const nOfVec);


/**
   Single precision sparse matrix vector product using MKL SCSRMV,
   with matrix stored in Compressed Sparse Row format (CSR).

   @param y       Output vector y (result of matrix-vector product)
   @param values  Vector containing the nnz elements of sparse matrix
   @param columns Columns vector
   @param rows    Rows vector
   @param x       Input vector x
   @param b       Bandwidth size (nnz for each row)
   @param nOfVec  Number of vectors
 */
void sparseMatrixMatrixComputationCSR( float               * const y,
                                       float         const * const values,
                                       int           const * const columns,
                                       int           const * const rows,
                                       float         const * const x,
                                       unsigned int          const n,
                                       unsigned int          const nOfVec);


/**
 * COMPUTESUBDISTSPARSE: Custom implementation of nostationary 3RHS
 * code using CSR.
 */
void computeSubDistSparse( double       *       Fattr,
                           double       * const Y,
                           double const * const p_sp,
                           int          *       ir,
                           int          *       jc,
                           int    const         n,   
                           int    const         d);

/**
 * COMPUTESUBDISTSPARSE: Custom implementation of stationary 1RHS
 * code using CSR.
 */
void computeSubDistSparse_spmv( double       *       Fattr,
                                double       * const Y,
                                double const * const p_sp,
                                int          *       ir,
                                int          *       jc,
                                int    const         n,   
                                int    const         d);


#endif

/* **********************************************************************
 *
 * AUTHORS
 *
 *   Dimitris Floros                         fcdimitr@auth.gr
 *
 * VERSION
 *
 *   0.1 - July 19, 2018
 *
 * CHANGELOG
 *
 *   0.1 (Jul 19, 2018) - Dimitris
 *       * initial implementation
 *
 * ********************************************************************** */
