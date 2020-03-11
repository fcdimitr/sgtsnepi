/*!
  \file   test_modules.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-28
*/


#include <iostream>
#include <random>

#include <cilk/cilk.h>
#include "utils.hpp"
#include "qq.hpp"
#include "pq.hpp"
#include "../csb/csb_wrapper.hpp"
#include <algorithm>    // std::random_shuffle

#define N_NUM 3
#define D_NUM 3
#define H_NUM 3

double * generateRandomCoord( int n, int d ){

  double *y = (double *) malloc( n*d*sizeof(double) );

  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine re;
  
  for (int i=0; i<n*d; i++)
    y[i] = unif(re);

  return y;
  
}

sparse_matrix *generateRandomCSC(int n){

  sparse_matrix *P = (sparse_matrix *) malloc(sizeof(sparse_matrix));
  
  P->n = n; P->m = n;
  
  P->col = (matidx *) malloc( (n+1)*sizeof(matidx) );

  for (int j=0 ; j<n ; j++)
    P->col[j] = rand() % 10 + 2;

  int cumsum = 0;
  for(int i = 0; i < P->n; i++){     
    int temp = P->col[i];
    P->col[i] = cumsum;
    cumsum += temp;
  }
  P->col[P->n] = cumsum;
  P->nnz = cumsum;

  P->row = (matidx *) malloc( (P->nnz)*sizeof(matidx) );
  P->val = (matval *) malloc( (P->nnz)*sizeof(matval) );

  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine re;

  int arr[n];
  for(int i = 0; i < n; ++i)
    arr[i] = i;
  
  for (int j=0 ; j<n ; j++) {

    std::random_shuffle(arr, arr+n);
    int k = 0;
    for (matidx l = P->col[j]; l < P->col[j+1]; l++){
      if (arr[k] == j) k++;
      P->row[l] = arr[k++];
      P->val[l] = unif(re);
    }

  }
  
  return P;
  
}

sparse_matrix *copySparseMatrix(sparse_matrix *P){

  sparse_matrix *P2 = (sparse_matrix *) malloc(sizeof(sparse_matrix));
  
  P2->n   = P->n;
  P2->m   = P->m;
  P2->nnz = P->nnz;
  
  P2->col = (matidx *) malloc( ((P2->n)+1)*sizeof(matidx) );
  P2->row = (matidx *) malloc( (P2->nnz)*sizeof(matidx) );
  P2->val = (matval *) malloc( (P2->nnz)*sizeof(matval) );

  for (int l = 0; l < P2->nnz; l++){
    P2->row[l] = P->row[l];
    P2->val[l] = P->val[l];
  }

  for (int l = 0; l < P2->n + 1; l++){
    P2->col[l] = P->col[l];
  }
  
  return P2;
  
}

void triu2(sparse_matrix *P){

  matidx n = P->n;
  matidx *Ap = P->col, *Ai = P->row;
  matval *Ax = P->val;
  
  for (matidx j=0 ; j<n ; j++) {

    for (matidx l = Ap[j]; l < Ap[j+1]; l++)
      if (Ai[l] < j) Ax[l] = 0;

  }
  
}

bool testEqual( sparse_matrix *P, sparse_matrix *P2 ){

  bool isEqual = true;
  
  for (matidx i=0; i < P->n+1; i++)
    if (P->col[i] != P2->col[i]){
      // std::cout << "COL: "<< i << " not equal!" << std::endl;
      isEqual = false;
    }

  for (matidx i=0; i < P->nnz; i++)
    if (P->row[i] != P2->row[i]){
      // std::cout << "ROW: " << i << " not equal!" << std::endl;
      isEqual = false;
    }

  for (matidx i=0; i < P->nnz; i++)
    if (P->val[i] != P2->val[i]){
      // std::cout << "VAL: " << i << " not equal!" << std::endl;
      isEqual = false;
    }

  return isEqual;
  
}

bool testAttractiveTerm( int n, int d){

  bool flag = true;

  double *y  = generateRandomCoord( n, d );
  sparse_matrix *P = generateRandomCSC(n);

  if( isSymPattern(P) )
    std::cout << "Input symmetric??" << std::endl;

  if( isSymValues(P) )
    std::cout << "Input symmetric values??" << std::endl;

  
  symmetrizeMatrix( P );

  if( !isSymPattern(P) )
    std::cout << "Not symmetric" << std::endl;

  if( !isSymValues(P) )
    std::cout << "Not symmetric values" << std::endl;

  sparse_matrix *P2 = copySparseMatrix(P);

  if( !testEqual(P, P2) )
    std::cout << "Symmetrized not equal!!!" << std::endl;

  if( !isSymPattern(P2) )
    std::cout << "Not symmetric pattern" << std::endl;

  if( !isSymValues(P2) )
    std::cout << "Not symmetric value" << std::endl;

  triu2( P2 );

  if( !isSymPattern(P2) )
    std::cout << "Not symmetric pattern" << std::endl;

  if( isSymValues(P2) )
    std::cout << "Still symmetric values?" << std::endl;

  symmetrizeMatrixWithSymPat( P2 ); 

  if( !isSymPattern(P2) )
    std::cout << "Symmetric pattern error!!!" << std::endl;

  if( !testEqual(P, P2) )
    std::cout << "Symmetrized not equal!!!" << std::endl;
  
  
  double *Fg = (double *) calloc( n*d , sizeof(double) );
  double *Ft = (double *) calloc( n*d , sizeof(double) );

  pq( Fg, y, P2->val, P2->row, P2->col, n, d);
  
  // initialize CSB object
  BiCsb<matval, matidx> *csb = NULL;

  // build CSB object (with default workers & BETA)
  csb = prepareCSB<matval, matidx>
    ( P->val, P->row, P->col,
      P->nnz,
      P->m,
      P->n,
      0, 0 );

  csb_pq( NULL, NULL, csb, y, Ft, n, d, 0, 0, 0 );

  double maxErr = 0;
  for (int i = 0; i<n*d; i++)
    maxErr = maxErr < abs( Fg[i] - Ft[i] )
                      ? abs( Fg[i] - Ft[i] )
                      : maxErr;

  if ( maxErr > 1e-10 )
    flag = false;
  
  deallocate(csb);
  free( P );
  free_sparse_matrix( P2 );
  free(y);
  free(Fg);
  free(Ft);
  
  return flag;
  
}

bool testRepulsiveTerm( int n, int d, int np){

  bool flag = true;

  double *y  = generateRandomCoord( n, d );
  double *Fg = (double *) calloc( n*d , sizeof(double) );
  double *Ft = (double *) malloc( n*d * sizeof(double) );

  double h[H_NUM] = {0.05, 0.08, 0.13};

  double zg = computeFrepulsive_exact(Fg, y, n, d);
  
  for (int i = 0; i<H_NUM; i++){

    Ft[0:(n*d)] = 0.0;
    double zt = computeFrepulsive_interp(Ft, y, n, d, h[i], np);

    double maxErr = 0;
    for (int jj = 0; jj<n*d; jj++)
      maxErr = maxErr < abs( Fg[jj] - Ft[jj] )
                      ? abs( Fg[jj] - Ft[jj] )
                      : maxErr;

    if ( maxErr > 1e-7 || abs(zg - zt)/zg > 1e-4 )
      flag = false;
    
  }
      
  free(y);
  free(Fg);
  free(Ft);

  return flag;
  
}


int main(void)
{

  int n[N_NUM] = {1000, 2000, 3000};
  int d[D_NUM] = {1,2,3};

  std::cout << "\n\n *** TESTING SG-TSNE-PI INSTALLATION ***\n\n";

  int np = getWorkers();
  
  std::cout << " %% Using " << np << " threads\n\n";
  
  std::cout << "\n - Attractive term [PQ]\n";

  int n_pass = 0;
  int n_fail = 0;
  
  for (int i = 0; i < N_NUM; i++){
    for (int j = 0; j < D_NUM; j++){
      std::cout << "   > N = " << n[i] << " D = " << d[j] << "..." << std::flush;

      bool status = testAttractiveTerm(n[i], d[j]);
      n_pass +=  status;
      n_fail += !status;
      
      if ( status )
        std::cout << "PASS" << std::endl;
      else
        std::cout << "FAIL!!!" << std::endl;
    }
  }

  std::cout << "\n - Repulsive term [QQ]\n";

  for (int i = 0; i < N_NUM; i++){
    for (int j = 0; j < D_NUM; j++){
      std::cout << "   > N = " << n[i] << " D = " << d[j] << "..." << std::flush;

      bool status = testRepulsiveTerm(n[i], d[j], np);
      n_pass +=  status;
      n_fail += !status;

      if ( status )
        std::cout << "PASS" << std::endl;
      else
        std::cout << "FAIL!!!" << std::endl;
    }
  }

  std::cout << "\n\n *** SUMMARY ***\n";
  std::cout << "  > " << n_pass << " tests passed" << std::endl;
  std::cout << "  > " << n_fail << " tests failed" << std::endl;

  if (n_fail == 0){
    std::cout << "\n *** INSTALLATION SUCESSFUL ***" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << "\n *** INSTALLATION FAILED!!! ***" << std::endl << std::endl;
    return -1;
  }
  
}
