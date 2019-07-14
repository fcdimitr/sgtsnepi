#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mat.h"

#include "cs.hpp"

#include "matfile.hpp"

void* safe_malloc(size_t n, char *name) {
  void *p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "Fatal: failed to allocate %zd bytes for %s.\n",
            n, name);
    exit(1);
  }
  return p;
}

cs *calcSimMat( cs *C, const double perplexity ) {

  // get transpose
  cs *Ct = cs_transpose( C, -1 );
  
  // get symmetrized kNN graph
  cs *Csym = cs_add( C, Ct, 0.5, 0.5);
  
  cs_spfree( Ct );
  
  return Csym;
    
}

cs *generateSparseMatrix( int *row, int *col, double *val,
                          int datasize, int knn, int flagSym ){

  int nnz = datasize * knn;
  
  // prepare sparse matrix using suite-sparse
  cs *C  = cs_spalloc (datasize, datasize, nnz, 1, 1) ;

  printf("FILLING CS MATRIX\n");
  
  for (int i=0; i<nnz; i++) {
    // printf( "(%d, %d): %f\n", row[i], col[i], val[i] );
    cs_entry( C, row[i], col[i], val[i] );
  }

  printf("COMPRESSING CS MATRIX\n");
  
  cs *Cc = cs_compress( C );

  cs_spfree( C );

  if (flagSym) {                // ----- SYMMETRIZED KNN

    printf("Building symmetrized kNN graph\n");
    cs *Cf = calcSimMat( Cc, 0 );
    cs *cs_spfree( Cc );
    return Cf;
    
  } else {                      // ----- ORIGINAL KNN

    printf("Using non-symmetric kNN graph\n");
    return Cc;
    
  }
  
}

int readMATdata( cs **C, int **perm,
                 double **lhsgold, double **rhsgold,
                 double *perplexity,
                 const char* basepath, const char *dataset,
                 const long long datasize, const long long knn,
                 const char* permName, const int VERSION,
                 const int flagSym ) {

  // keep rows, cols, and values of matrix
  int *row, *col;
  double *val;
  
  // prepare buffer for MAT-file name
  char matName[100];
  char permStructName[100];

  // MAT-file variable
  MATFile *pmat;

  // mxArray pointers for reading through MAT-file
  mxArray *kidx, *kdist, *mxPermList, *mxPerm;
  int *kidx_ptr, *perm_ptr;
  double * kdist_ptr;
  
  // update MAT-file name
  sprintf(matName, "%s/%s_%d.mat",
          basepath, dataset, datasize);

  // update perms struct name
  sprintf( permStructName, "iPerm_k%d", knn );
  
  // open MAT-file
  pmat = matOpen( matName, "r" );

  // vaidate that MAT-file opened, otherwise return non-zero code
  if (pmat == NULL) {
    printf("Error opening file %s\n", matName);
    return(1);
  } else {
    printf("Opened file %s\n", matName);
  }

  // read kNN indexes
  kidx       = matGetVariable( pmat, "IDX" );
  kdist      = matGetVariable( pmat, "D" );

  // Size of matrix
  const mwSize *matSize = mxGetDimensions(kidx);

  // check that k is smaller that stored k
  if (knn > matSize[1]){
    printf("k=%d too large!\n", knn);
    return(1);
  }
  
  mxPermList = matGetVariable( pmat, permStructName );
  printf("Name struct: %s\n", permStructName);
  
  if ( mxPermList == NULL ) {
    
    printf("Permutation struct %s was not found!\n", permStructName);
    mxPerm = NULL;
    
  } else {

    printf("Perm name: %s", permName);
    mxPerm     = mxGetField( mxPermList, 0, permName );
    printf(" read\n");
    
  }
  
  if (mxPerm == NULL){
    
    printf("Permutation %s was not found!\n", permName);

    *perm = NULL;
    
  } else {

    printf("Permutation %s ", permName);
    
    perm_ptr = (int *) mxGetData( mxPerm );
    *perm    = (int *) safe_malloc( datasize * sizeof(int), "perm" );
    for (int i = 0; i < datasize; i++){
      perm[0][ i ] = perm_ptr[ i ] - 1;
    }
    mxDestroyArray( mxPermList );
    
    printf("copied\n");
  }

  printf("Sparse matrix ");
  // pass data to pointers
  kidx_ptr  = (int *)    mxGetData( kidx   );
  kdist_ptr = (double *) mxGetData( kdist  );

  // printf( "KIDX size: [%dx%d]\n", matSize[0], matSize[1] );

  printf( "Size to allocate %zu (%d - %d - %d)\n",
          datasize * knn * sizeof(int),
          datasize, knn, sizeof(int) );
  
  row  = (int *)    safe_malloc( datasize * knn * sizeof(int)   , "row" );
  col  = (int *)    safe_malloc( datasize * knn * sizeof(int)   , "col" );
  val  = (double *) safe_malloc( datasize * knn * sizeof(double), "val" );
  
  for (int j = 0; j < knn; j++)
    for (int i = 0; i < datasize; i++){
      row[ i + j*datasize ] = i;
      col[ i + j*datasize ] = kidx_ptr[ i + j*datasize ] - 1;
    }
  
  memcpy( val, kdist_ptr, datasize*knn*sizeof(double) );

  printf("copied\n");
  
#ifdef VERIFY
  // --------------------------------------------------
  // Check for existence of ground truth
  
  int dim;

  // prepare buffer for variable LHS
  char lhsName[100];
  
  switch (VERSION) {
    
  case CODE_VERSION_S1:
    sprintf( lhsName, "lhs1_test_s_k%d", knn );
    dim = 1;
    break;

  case CODE_VERSION_S3:
    sprintf( lhsName, "lhs3_test_s_k%d", knn );
    dim = 3;
    break;

  case CODE_VERSION_NS3:
    sprintf( lhsName, "lhs3_test_ns_k%d", knn );
    dim = 3;
    break;

  default:
    printf("Unknown version %d\n", VERSION);
    break;

  }

  // -------------------- RHS
  
  mxArray *mx_rhsgold = matGetVariable( pmat, "rhs3_test" );

  if (mx_rhsgold == NULL) {
    printf( "Ground truth data unavailable\n" );
    rhsgold[0] = NULL;
  } else {
    double *mx_rhsgold_data = (double *) mxGetData( mx_rhsgold );
    rhsgold[0] = (double *) safe_malloc( datasize * dim * sizeof(double),
                                         "rhs" );
    
    for (int i = 0; i < datasize; i++)
      for (int j = 0; j < dim; j++)
        rhsgold[0][ j + i*dim ] = mx_rhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_rhsgold );
  }

  // -------------------- LHS
  
  mxArray *mx_lhsgold = matGetVariable( pmat, lhsName );

  if (mx_lhsgold == NULL) {
    printf( "Ground truth data unavailable\n" );
    lhsgold[0] = NULL;
  } else {
    double *mx_lhsgold_data = (double *) mxGetData( mx_lhsgold );
    lhsgold[0] = (double *) safe_malloc( datasize * dim * sizeof(double),
                                         "lhs" );

    for (int i = 0; i < datasize; i++)
      for (int j = 0; j < dim; j++)
        lhsgold[0][ j + i*dim ] = mx_lhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_lhsgold );
  }

  // -------------------- check if perplexity exists
  mxArray *mx_perpl = matGetVariable( pmat, "perplexity" );

  if (mx_perpl == NULL){
    printf( "Unknown perplexity\n" );
    perplexity[0] = 0;
  } else {
    perplexity[0] = mxGetScalar( mx_perpl );
    printf( "Perplexity = %4.2f\n", perplexity[0] );
    mxDestroyArray( mx_perpl );
  }

#else
  rhsgold[0] = NULL;
  lhsgold[0] = NULL;
  perplexity[0] = 0;
  printf( "Not verifying results\n" );
#endif

  // build sparse matrix
  C[0] = generateSparseMatrix( row, col, val, datasize, knn, flagSym );

  // free unecessary variables
  free( row );
  free( val );
  free( col );
  printf("Buffers freed\n");
  
  // destroy array read
  mxDestroyArray( kidx   );
  mxDestroyArray( kdist  );

  // close MAT-file -- otherwise error
  if (matClose(pmat) != 0) {
    printf("Error closing file %s\n", matName);
    return(1);
  } else {
    printf("Closed file %s\n", matName);
  }

  return 0;
  
}
