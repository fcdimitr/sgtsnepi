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

  delete [] P->row;
  delete [] P->col;
  delete [] P->val;

}

uint32_t makeStochastic(sparse_matrix P){

  int *stoch = new int [P.n] ();
  
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

  delete [] stoch;

  return nStoch;
  
}

void symmetrizeMatrix( sparse_matrix *P ){

  // Get sparse matrix
  matidx* row_P = P->col;
  matidx* col_P = P->row;
  matval* val_P = P->val;

  matidx N = P->n;
  
  // Count number of elements and row counts of symmetric matrix
  int* row_counts = new int [N] ();
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
  matidx* sym_row_P = new matidx [N+1];
  matidx* sym_col_P = new matidx [no_elem];
  matval* sym_val_P = new matval [no_elem];
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
  int* offset = new int [N] ();
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
  delete [] P->row; P->row = sym_col_P;
  delete [] P->col; P->col = sym_row_P;
  delete [] P->val; P->val = sym_val_P;

  P->nnz = no_elem;
  
  // Free up some memery
  delete [] offset; offset = NULL;
  delete [] row_counts; row_counts  = NULL;

  return;
  
}


void permuteMatrix( sparse_matrix *P, int *perm, int *iperm ){

  // Get sparse matrix
  matidx* row_P = P->row;
  matidx* col_P = P->col;
  matval* val_P = P->val;

  int N = P->n; matidx nnz = P->nnz;
  
  // Allocate memory for permuted matrix
  matidx* perm_row_P = new matidx [nnz];
  matidx* perm_col_P = new matidx [N+1];
  matval* perm_val_P = new matval [nnz];
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
  delete [] P->row; P->row = perm_row_P;
  delete [] P->col; P->col = perm_col_P;
  delete [] P->val; P->val = perm_val_P;

  return;
  
}


void printSparseMatrix( sparse_matrix P ){

  std::cout << "m = " << P.m
            << " | n = " << P.n
            << " | nnz = " << P.nnz
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

