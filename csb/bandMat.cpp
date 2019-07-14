#include "bandMat.hpp"


// ==================================================
// === UTILITIES

void transformCSCtoCSR(double       * const acsr,
                       int          * const ja,
                       int          * const ia,
                       double       * const acsc,
                       int          * const ja1,
                       int          * const ia1,
                       unsigned int   const n) {

  MKL_INT job[8] = {1, 0, 0, 0, 0, 1, 0, 0};

  MKL_INT m = (MKL_INT) n;
  
  MKL_INT * info = new MKL_INT[1];
  
  mkl_dcsrcsc (job  , &m  ,
               acsr , ja  , ia,
               acsc , ja1 , ia1 , info );
  
}

void transformCSCtoCSR(float        * const acsr,
                       int          * const ja,
                       int          * const ia,
                       float        * const acsc,
                       int          * const ja1,
                       int          * const ia1,
                       unsigned int   const n) {

  MKL_INT job[8] = {1, 0, 0, 0, 0, 1, 0, 0};

  MKL_INT m = (MKL_INT) n;

  MKL_INT * info = new MKL_INT[1];
  
  mkl_scsrcsc (job  , &m  ,
               acsr , ja  , ia,
               acsc , ja1 , ia1 , info );
  
}


// ==================================================
// === MATRIX-VECTOR PRODUCT ROUTINES


// --------------------------------------------------
// ---------- ?GBMV: BANDED MATRIX VECTOR

void bandMatComputation( double              * const y,
                         double        const * const A,
                         double        const * const x,
                         unsigned int          const n,
                         unsigned int          const b,
                         unsigned int          const nOfVec){

  unsigned int kl = (b - 1) / 2;

  for ( int jj = 0; jj < nOfVec; jj++ ) {

    cblas_dgbmv(CblasColMajor, CblasNoTrans, n, n, kl, kl,
                1, A, b, &(x[jj*n]), 1, 0, &(y[jj*n]), 1);
  }
  
}

void bandMatComputation( float               * const y,
                         float         const * const A,
                         float         const * const x,
                         unsigned int          const n,
                         unsigned int          const b,
                         unsigned int          const nOfVec){

  unsigned int kl = (b - 1) / 2;

  for ( int jj = 0; jj < nOfVec; jj++ ) {

    cblas_sgbmv(CblasColMajor, CblasNoTrans, n, n, kl, kl,
                1, A, b, &(x[jj*n]), 1, 0, &(y[jj*n]), 1);

  }
  
}


// --------------------------------------------------
// ---------- ?CSCMV: MATRIX VECTOR PRODUCT USING SPARSE CSC

void sparseComputation( double              * const y,
                        double        const * const values,
                        int           const * const rows,
                        int           const * const columns,
                        double        const * const x,
                        unsigned int          const n,
                        unsigned int          const nOfVec){

  char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
  char transa = 'n';
  double alpha = 1;
  double beta  = 0;

  MKL_INT m = n;
  MKL_INT k = n;

  for ( int jj = 0; jj < nOfVec; jj++)
  
    mkl_dcscmv(&transa, &m, &k, &alpha, matdescra, values, rows,
               columns,  &(columns[1]), &(x[jj*n]), &beta, &(y[jj*n]));
  
}

void sparseComputation( float               * const y,
                        float         const * const values,
                        int           const * const rows,
                        int           const * const columns,
                        float         const * const x,
                        unsigned int          const n,
                        unsigned int          const nOfVec){

  char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
  char transa = 'n';
  float alpha = 1;
  float beta  = 0;

  MKL_INT m = n;
  MKL_INT k = n;

  for ( int jj = 0; jj < nOfVec; jj++)
  
    mkl_scscmv(&transa, &m, &k, &alpha, matdescra, values, rows,
               columns,  &(columns[1]), &(x[jj*n]), &beta, &(y[jj*n]));

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


// --------------------------------------------------
// ---------- ?SBMV: SYMMETRIC BANDED MATRIX VECTOR

void symBandMatComputation( double              * const y,
                            double        const * const A,
                            double        const * const x,
                            unsigned int          const n,
                            unsigned int          const b,
                            unsigned int          const nOfVec){

  for ( int jj = 0; jj < nOfVec; jj++)
  
    cblas_dsbmv(CblasColMajor, CblasUpper, n, b-1,
                1, A, b, &(x[jj*n]), 1, 0, &(y[jj*n]), 1);

}

void symBandMatComputation( float               * const y,
                            float         const * const A,
                            float         const * const x,
                            unsigned int          const n,
                            unsigned int          const b,
                            unsigned int          const nOfVec){

  for ( int jj = 0; jj < nOfVec; jj++)
    
    cblas_ssbmv(CblasColMajor, CblasUpper, n, b-1,
                1, A, b, &(x[jj*n]), 1, 0, &(y[jj*n]), 1);

}


// --------------------------------------------------
// ---------- ?CSRSYMV: SYMMETRICAL MATRIX VECTOR PRODUCT

void symSparseComputation( double              * const y,
                           double        const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           double        const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec){

  char uplo[1] = {'u'};
  
  MKL_INT m = n;

  for ( int jj = 0; jj < nOfVec; jj++)
  
    mkl_cspblas_dcsrsymv(uplo, &m, values, rows, columns, &(x[jj*n]), &(y[jj*n]));
  
}

void symSparseComputation( float               * const y,
                           float         const * const values,
                           int           const * const rows,
                           int           const * const columns,
                           float         const * const x,
                           unsigned int          const n,
                           unsigned int          const nOfVec){

  char uplo[1] = {'u'};
    
  MKL_INT m = n;

  for ( int jj = 0; jj < nOfVec; jj++)
  
    mkl_cspblas_scsrsymv(uplo, &m, values, rows, columns, &(x[jj*n]), &(y[jj*n]));
  
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

// --------------------------------------------------
// --- CSC MATRIX-MATRIX PRODUCT

void sparseMatrixMatrixComputation( double              * const y,
                                    double        const * const values,
                                    int           const * const rows,
                                    int           const * const columns,
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
    
  mkl_dcscmm (transa, &m, &nVecs, &k, &alpha, matdescra, values, rows,
              columns,  &(columns[1]), x, &nVecs, &beta, y, &nVecs );

}

void sparseMatrixMatrixComputation( float               * const y,
                                    float         const * const values,
                                    int           const * const rows,
                                    int           const * const columns,
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
  
  mkl_scscmm (transa, &m, &nVecs, &k, &alpha, matdescra, values, rows,
              columns,  &(columns[1]), x, &nVecs, &beta, y, &nVecs );

}
