#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mat.h"
#include <iostream>

#include "cs.hpp"

#include "matfileio.hpp"

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

cs *generateSparseMatrix( CS_INT *row, CS_INT *col, double *val,
                          CS_INT datasize, CS_INT knn, CS_INT flagSym ){

  CS_INT nnz = ( (CS_INT) datasize ) * ( (CS_INT) knn );
  
  // prepare sparse matrix using suite-sparse
  cs *C  = cs_spalloc( (CS_INT) datasize,
                       (CS_INT) datasize,
                       nnz,
                       1, 1) ;

  std::cout << "FILLING CS MATRIX\n" << std::endl;
  
  for (CS_INT i=0; i<nnz; i++) {

    cs_entry( C,
              (CS_INT) row[i],
              (CS_INT) col[i],
              (double)    val[i] );
    
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

CS_INT readMATdata( cs **C, CS_INT **perm,
                 double **lhss1gold,
                 double **lhss3gold,
                 double **lhsns3gold,
                 double **rhsgold1,
                 double **rhsgold3,
                 double *perplexity,
                 const char* basepath, const char *dataset,
                 const CS_INT datasize, const CS_INT knn,
                 const char* permName,
                 const CS_INT flagSym ) {

  // keep rows, cols, and values of matrix
  CS_INT *row, *col;
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
    *perm    = (CS_INT *) safe_malloc( datasize * sizeof(CS_INT), "perm" );
    for (CS_INT i = 0; i < datasize; i++){
      perm[0][ i ] = (CS_INT) perm_ptr[ i ] - 1;
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
          datasize * knn * sizeof(CS_INT),
          datasize, knn, sizeof(CS_INT) );
  
  row  = (CS_INT *)    safe_malloc( datasize * knn * sizeof(CS_INT)   , "row" );
  col  = (CS_INT *)    safe_malloc( datasize * knn * sizeof(CS_INT)   , "col" );
  val  = (double *) safe_malloc( datasize * knn * sizeof(double), "val" );
  
  for (CS_INT j = 0; j < knn; j++)
    for (CS_INT i = 0; i < datasize; i++){
      row[ i + j*datasize ] = i;
      col[ i + j*datasize ] = (CS_INT) kidx_ptr[ i + j*datasize ] - 1;
    }
  
  memcpy( val, kdist_ptr, datasize*knn*sizeof(double) );

  printf("copied\n");
  
#ifdef VERIFY
  // --------------------------------------------------
  // Check for existence of ground truth
  
  char lhsNameS1[100];
  char lhsNameS3[100];
  char lhsNameNS3[100];
  
  sprintf( lhsNameS1,  "lhs1_test_s_k%d", knn );

  sprintf( lhsNameS3,  "lhs3_test_s_k%d", knn );

  sprintf( lhsNameNS3, "lhs3_test_ns_k%d", knn );

  // -------------------- RHS
  
  mxArray *mx_rhsgold = matGetVariable( pmat, "rhs3_test" );

  if (mx_rhsgold == NULL) {
    printf( "RHS -- Ground truth data unavailable\n" );
    rhsgold1[0] = NULL;
    rhsgold3[0] = NULL;
  } else {
    printf( "RHS -- " );
    double *mx_rhsgold_data = (double *) mxGetData( mx_rhsgold );
    rhsgold1[0] = (double *) safe_malloc( datasize * sizeof(double),
                                          "rhs" );

    rhsgold3[0] = (double *) safe_malloc( datasize * 3 * sizeof(double),
                                          "rhs" );
    
    for (CS_INT i = 0; i < datasize; i++)
      for (CS_INT j = 0; j < 1; j++)
        rhsgold1[0][ j + i ] = mx_rhsgold_data[ i + j*datasize ];

    for (CS_INT i = 0; i < datasize; i++)
      for (CS_INT j = 0; j < 3; j++)
        rhsgold3[0][ j + i*3 ] = mx_rhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_rhsgold );
    printf( "READ\n");
  }

  // -------------------- LHS

  mxArray *mx_lhsgold;
  
  mx_lhsgold = matGetVariable( pmat, lhsNameS1 );

  if (mx_lhsgold == NULL) {
    printf( "LSH -- %s -- Ground truth data unavailable\n", lhsNameS1 );
    lhss1gold[0] = NULL;
  } else {
    printf( "LSH -- %s -- ", lhsNameS1 );
    double *mx_lhsgold_data = (double *) mxGetData( mx_lhsgold );
    lhss1gold[0] = (double *) safe_malloc( datasize * 1 * sizeof(double),
                                           "lhs" );

    for (CS_INT i = 0; i < datasize; i++)
      for (CS_INT j = 0; j < 1; j++)
        lhss1gold[0][ j + i ] = mx_lhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_lhsgold );
    printf( "READ\n");
  }

  mx_lhsgold = matGetVariable( pmat, lhsNameS3 );

  if (mx_lhsgold == NULL) {
    printf( "LSH -- %s -- Ground truth data unavailable\n", lhsNameS3 );
    lhss3gold[0] = NULL;
  } else {
    printf( "LSH -- %s -- ", lhsNameS3 );
    double *mx_lhsgold_data = (double *) mxGetData( mx_lhsgold );
    lhss3gold[0] = (double *) safe_malloc( datasize * 3 * sizeof(double),
                                           "lhs" );

    for (CS_INT i = 0; i < datasize; i++)
      for (CS_INT j = 0; j < 3; j++)
        lhss3gold[0][ j + i*3 ] = mx_lhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_lhsgold );
    printf( "READ\n");
  }

  mx_lhsgold = matGetVariable( pmat, lhsNameNS3 );

  if (mx_lhsgold == NULL) {
    printf( "LSH -- %s -- Ground truth data unavailable\n", lhsNameNS3 );
    lhsns3gold[0] = NULL;
  } else {
    printf( "LSH -- %s -- ", lhsNameNS3 );
    double *mx_lhsgold_data = (double *) mxGetData( mx_lhsgold );
    lhsns3gold[0] = (double *) safe_malloc( datasize * 3 * sizeof(double),
                                           "lhs" );

    for (CS_INT i = 0; i < datasize; i++)
      for (CS_INT j = 0; j < 3; j++)
        lhsns3gold[0][ j + i*3 ] = mx_lhsgold_data[ i + j*datasize ];

    mxDestroyArray( mx_lhsgold );
    printf( "READ\n");
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
  lhss1gold[0]  = NULL;
  lhss3gold[0]  = NULL;
  lhsns3gold[0] = NULL;
  rhsgold1[0]   = NULL;
  rhsgold3[0]   = NULL; 
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
