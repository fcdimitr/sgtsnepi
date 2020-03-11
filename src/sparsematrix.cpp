/*!
  \file   sparsematrix.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-20
*/

#include "sparsematrix.hpp"

#include <cmath>
#include <limits>
#include <iostream>
#include <cilk/cilk.h>
#include <fstream>

void free_sparse_matrix(sparse_matrix * P){

  free(P->row);
  free(P->col);
  free(P->val);
  
}

uint32_t makeStochastic(sparse_matrix P){

  int *stoch = static_cast< int *>( calloc(P.n, sizeof(int)) );
  
  cilk_for (int j=0; j<P.n; j++){

    double sum = 0;

    for ( uint32_t t = P.col [j] ; t < P.col[j+1] ; t++) 
      sum += P.val[t];
    
    if ( std::abs(sum - 1) > 1e-12 )
      for ( uint32_t t = P.col [j] ; t < P.col [j+1] ; t++) 
        P.val[t] /= sum;
    else
      stoch[j] = 1;
    
  }

  uint32_t nStoch = 0;
  
  for (int j=0; j<P.n; j++){
    nStoch += stoch[j];
  }

  free( stoch );

  return nStoch;
  
}

void symmetrizeMatrix( sparse_matrix *P ){

  // Get sparse matrix
  matidx* row_P = P->col;
  matidx* col_P = P->row;
  matval* val_P = P->val;

  matidx N = P->n;
  
  // Count number of elements and row counts of symmetric matrix
  int* row_counts = (int*) calloc(N, sizeof(int));
  if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(matidx n = 0; n < N; n++) {
    for(matidx i = row_P[n]; i < row_P[n + 1]; i++) {
      
      // Check whether element (col_P[i], n) is present
      bool present = false;
      for(matidx m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
	if(col_P[m] == n) present = true;
      }
      if(present) row_counts[n]++;
      else {
        row_counts[n]++;
        row_counts[col_P[i]]++;
      }
    }
  }
  int no_elem = 0;
  for(matidx n = 0; n < N; n++) no_elem += row_counts[n];
  
  // Allocate memory for symmetrized matrix
  matidx* sym_row_P = (matidx*) malloc((N + 1) * sizeof(matidx));
  matidx* sym_col_P = (matidx*) malloc(no_elem * sizeof(matidx));
  matval* sym_val_P = (matval*) malloc(no_elem * sizeof(matval));
  if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  
  // Construct new row indices for symmetric matrix
  sym_row_P[0] = 0;
  for(matidx n = 0; n < N; n++) {
    sym_row_P[n + 1] = sym_row_P[n] + (matidx) row_counts[n];
  }
  
  // Fill the result matrix
  int* offset = (int*) calloc(N, sizeof(int));
  if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(matidx n = 0; n < N; n++) {
    for(matidx i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])
      
      // Check whether element (col_P[i], n) is present
      bool present = false;
      for(matidx m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
	if(col_P[m] == n) {
	  present = true;
	  if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
	    sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
	    sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
	    sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
	    sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
	  }
	}
      }
      
      // If (col_P[i], n) is not present, there is no addition involved
      if(!present) {
	sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
	sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
	sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
	sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
      }
      
      // Update offsets
      if(!present || (present && n <= col_P[i])) {
	offset[n]++;
	if(col_P[i] != n) offset[col_P[i]]++;               
      }
    }
  }
  
  // Return symmetrized matrices
  free(P->row); P->row = sym_col_P;
  free(P->col); P->col = sym_row_P;
  free(P->val); P->val = sym_val_P;

  P->nnz = no_elem;
  
  // Free up some memery
  free(offset); offset = NULL;
  free(row_counts); row_counts  = NULL;

  return;
  
}

bool isSymPattern( sparse_matrix *P ){

  matidx n = P->n; // square matrices

  matidx *Ap, *Ai;

  bool isSym = true;
  
  Ap = P->col; Ai = P->row;
  
  cilk_for (matidx j = 0 ;
            j < n ;
            j++){
    
    for (matidx p = Ap[j] ;
         p < Ap[j+1] && isSym ;
         p++) {

      matidx i = Ai[p];

      bool isSymVal = false;
      
      for (matidx q = Ap[i] ;
           q < Ap[i+1] && !isSymVal ;
           q++)

        if (Ai[q] == j) isSymVal = true;

      isSym &= isSymVal;
      
    }
    
  }

  return isSym;
  
}


bool isSymValues( sparse_matrix *P ){

  matidx n = P->n; // square matrices

  matidx *Ap, *Ai, j, i, p, q;
  matval *Ax;
  
  bool isSym = true;
  
  Ap = P->col; Ai = P->row;
  Ax = P->val;

  // loop through columns of matrix
  for (j = 0 ;
       j < n  && isSym ;
       j++){

    // loop through nonzero rows
    for (p = Ap[j] ;
         p < Ap[j+1] && isSym ;
         p++) {

      // (i,j) element
      i = Ai[p];

      bool isSymVal = false;

      // access column (:,i)
      for (q = Ap[i] ;
           q < Ap[i+1] && !isSymVal ;
           q++) {

        // (k,i) element
        matidx k = Ai[q];
      
        if (k == j &&
            abs( Ax[q] - Ax[p] ) < 1e-10 )
          isSymVal = true;

      } // for (q)
        
      isSym &= isSymVal;
      
    }
    
  }

  return isSym;
  
}


void symmetrizeMatrixWithSymPat( sparse_matrix *P ){

  matidx n = P->n; // square matrices

  matidx *Ap, *Ai;
  matval *Ax;

  Ap = P->col; Ai = P->row;
  Ax = P->val;

  // loop through columns of matrix
  cilk_for (matidx j = 0 ;
            j < n ;
            j++){

    // loop through nonzero rows
    for (matidx p = Ap[j] ;
         p < Ap[j+1];
         p++) {

      // i < j
      if (Ai[p] > j) continue;
      
      // (i,j) element
      matidx i = Ai[p];

      // access column (:,i)
      for (matidx q = Ap[i] ;
           q < Ap[i+1];
           q++){

        // (k,i) element
        matidx k = Ai[q];
        
        if (k == j) {
          Ax[q] += Ax[p];
          Ax[p]  = Ax[q];
        }

      } // for (q)

    }
    
  }

}


void permuteMatrix( sparse_matrix *P, int *perm, int *iperm ){

  // Get sparse matrix
  matidx* row_P = P->row;
  matidx* col_P = P->col;
  matval* val_P = P->val;

  int N = P->n; matidx nnz = P->nnz;
  
  // Allocate memory for permuted matrix
  matidx* perm_row_P = (matidx*) malloc( nnz    * sizeof(matidx));
  matidx* perm_col_P = (matidx*) malloc((N + 1) * sizeof(matidx));
  matval* perm_val_P = (matval*) malloc( nnz    * sizeof(matval));
  if(perm_row_P == NULL || perm_col_P == NULL || perm_val_P == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  
  // Construct new row indices for symmetric matrix
  int nz = 0, j; uint32_t t;
  for (int k = 0 ; k < N ; k++) {

    perm_col_P[k] = nz ;                   /* column k of C is column q[k] of A */
    j = perm[k];
    
    for (t = col_P [j] ; t < col_P [j+1] ; t++) {
      perm_val_P [nz] = val_P [t] ;  /* row i of A is row pinv[i] of C */
      perm_row_P [nz++] = iperm [row_P [t]];
    }
    
  }
  perm_col_P[N] = nz ; 
  
  // Return symmetrized matrices
  free(P->row); P->row = perm_row_P;
  free(P->col); P->col = perm_col_P;
  free(P->val); P->val = perm_val_P;

  return;
  
}


void printSparseMatrix( sparse_matrix P ){

  double hash = 0;
  for (int j = 0; j < P.nnz; j++)
    hash += P.val[j];
  
  std::cout << "m = " << P.m
            << "| n = " << P.n
            << "| nnz = " << P.nnz
            << "| total sum = " << hash
            << std::endl;
  
  if ( P.nnz < 150 )
  
    for (int j = 0; j < P.n; j++){
      
      int off = P.col[j];
      int nnzcol = P.col[j+1] - off;

      for (int idx = off; idx < (off + nnzcol); idx++ ){

        int i = P.row[idx];
        double v = P.val[idx];
      
        printf( " (%d,%d)   %.4f \n", i+1, j+1, v);
      
      }
    
    
    }

}

