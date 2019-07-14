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

  for (int l = 0; l < P->nnz; l++){
    P->row[l] = rand() % n;
    P->val[l] = unif(re);
  }
  
  return P;
  
}



bool testAttractiveTerm( int n, int d){

  bool flag = true;

  double *y  = generateRandomCoord( n, d );
  sparse_matrix *P = generateRandomCSC(n);
  symmetrizeMatrix( P );
 

  double *Fg = (double *) calloc( n*d , sizeof(double) );
  double *Ft = (double *) calloc( n*d , sizeof(double) );

  pq( Fg, y, P->val, P->row, P->col, n, d);
  
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
