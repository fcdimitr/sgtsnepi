/*!
  \file   sgtsne.cpp
  \brief  Entry point to SG-t-SNE

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.


  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <string>
#include <unistd.h>
#include <limits>
#include <cmath>

#include "types.hpp"
#include <string>
#include "sparsematrix.hpp"
#include "utils.hpp"
#include "gradient_descend.hpp"
#include "graph_rescaling.hpp"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include <metis.h>
#include "../csb/csb_wrapper.hpp"

// #define FLAG_BSDB_PERM

coord * sgtsne(sparse_matrix P, tsneparams params,
               coord *y_in,
               double **timeInfo)
{

  // ~~~~~~~~~~ unless h is specified, use default ones
  if (params.h <= 0)
    switch (params.d){
    case 1:
      params.h = 0.5;
      break;
    case 2:
      params.h = 0.7;
      break;
    case 3:
      params.h = 1.2;
      break;
    }
  
  // ~~~~~~~~~~ print input parameters
  printParams( params );

  // ~~~~~~~~~~ make sure input matrix is column stochastic
  uint32_t nStoch = makeStochastic( P );
  std::cout << nStoch << " out of " << P.n
            << " nodes already stochastic"
            << std::endl;
  
  // ~~~~~~~~~~ prepare graph for SG-t-SNE
  
  // ----- lambda rescaling
  if (params.lambda == 1)
    std::cout << "Skipping λ rescaling..." << std::endl;
  else
    lambdaRescaling( P, params.lambda, false, params.dropLeaf );
  
  // ----- symmetrizing
  symmetrizeMatrix( &P );

 
  // ----- normalize matrix (total sum is 1.0)
  double sum_P = .0;
  for(int i = 0; i < P.nnz; i++){
    sum_P += P.val[i];
  }
  for(int i = 0; i < P.nnz; i++) {
    P.val[i] /= sum_P;
  }

  
  // ~~~~~~~~~~ extracting BSDB permutation
  idx_t *perm = static_cast<idx_t *>( malloc(P.n * sizeof(idx_t)) );
  idx_t *iperm = static_cast<idx_t *>( malloc(P.n * sizeof(idx_t)) );

#ifdef FLAG_BSDB_PERM
  
  std::cout << "Nested dissection permutation..." << std::flush;
  // idx_t options[METIS_NOPTIONS];
  // METIS_SetDefaultOptions(options);
  // options[METIS_OPTION_NUMBERING] = 0;
  
  int status = METIS_NodeND( &P.n,
                             reinterpret_cast<idx_t *> (P.col), 
                             reinterpret_cast<idx_t *> (P.row),
                             NULL, NULL,
                             perm, iperm );


  permuteMatrix( &P, perm, iperm );
  
  
  if( status != METIS_OK ) {
    std::cerr << "METIS error."; exit(1);
  }

  std::cout << "DONE" << std::endl;

#else

  for( int i = 0; i < P.n; i++ ){
    perm[i]  = i;
    iperm[i] = i;
  }

#endif
  
  printSparseMatrix(P);


  // ~~~~~~~~~~ build CSB matrix

  // initialize CSB object
  BiCsb<matval, matidx> *csb = NULL;

  // build CSB object (with default workers & BETA)
  csb = prepareCSB<matval, matidx>
    ( P.val, P.row, P.col,
      P.nnz,
      P.m,
      P.n,
      0, 0 );

  // ~~~~~~~~~~ initial embedding coordinates
  
  coord *y = static_cast<coord *> ( malloc( params.n * params.d * sizeof(coord) ) );

  if (y_in == NULL){
    
    // ----- Initialize Y
    for(int i = 0; i < params.n*params.d; i++){
      y[i] = randn() * .0001;
    }
    
  } else {

    std::copy( y_in, y_in + params.n*params.d, y );
    
  }

  // ~~~~~~~~~~ gradient descent
  kl_minimization( y, params, csb, timeInfo );


  // ~~~~~~~~~~ inverse permutation
  coord *y_inv = static_cast<coord *> ( malloc( params.n * params.d * sizeof(coord) ) );

  for (int i=0; i<params.n; i++)
    for (int j=0; j<params.d; j++)
      y_inv[i*params.d + j] = y[ iperm[i]*params.d + j ];
  

  // ~~~~~~~~~~ dellocate memory
  
  deallocate(csb);
  free( y );
  free( perm );
  free( iperm );

  return y_inv;
  
}



void equalizeVertex(double*  val_P,
		    double * distances,
		    double perplexity,
		    int nn){

  bool found = false;
  double beta = 1.0;
  double min_beta = -std::numeric_limits<double>::max();
  double max_beta =  std::numeric_limits<double>::max();
  double tol = 1e-5;
  
  // Iterate until we found a good perplexity
  int iter = 0; double sum_P;
  while(!found && iter < 200) {
    
    // Compute Gaussian kernel row
    for(int m = 0; m < nn; m++) val_P[m] = exp(-beta * distances[m + 1]);
    
    // Compute entropy of current row
    sum_P = std::numeric_limits<double>::min();
    for(int m = 0; m < nn; m++) sum_P += val_P[m];
    double H = .0;
    for(int m = 0; m < nn; m++) H += beta * (distances[m + 1] * val_P[m]);
    H = (H / sum_P) + log(sum_P);
    
    // Evaluate whether the entropy is within the tolerance level
    double Hdiff = H - log(perplexity);
    if(Hdiff < tol && -Hdiff < tol) {
      found = true;
    }
    else {
      if(Hdiff > 0) {
	min_beta = beta;
	if(max_beta == std::numeric_limits<double>::max() || max_beta == -std::numeric_limits<double>::max())
	  beta *= 2.0;
	else
	  beta = (beta + max_beta) / 2.0;
      }
      else {
	max_beta = beta;
	if(min_beta == -std::numeric_limits<double>::max() || min_beta == std::numeric_limits<double>::max())
	  beta /= 2.0;
	else
	  beta = (beta + min_beta) / 2.0;
      }
    }
    
    // Update iteration counter
    iter++;
  }
  
  for(int m = 0; m < nn; m++) val_P[m] /= sum_P;
      
}


sparse_matrix perplexityEqualization( int *I, double *D, int n, int nn, double u ){

  sparse_matrix P;
  matval *val;
  matidx *row, *col;

  // allocate space for CSC format
  val = static_cast<matval *>( malloc( n*nn   * sizeof(matval)) );
  row = static_cast<matidx *>( malloc( n*nn   * sizeof(matidx)) );
  col = static_cast<matidx *>( calloc( (n+1) , sizeof(matidx)) );

  // perplexity-equalization of kNN input
  cilk_for(int i = 0; i < n; i++) {

    equalizeVertex( &val[i*nn], &D[i*(nn+1)], u, nn );
    
  }
  
  // prepare column-wise kNN graph
  int nz = 0;
  for (int j=0; j<n; j++){
    col[j] = nz;
    for (int idx=0; idx<nn; idx++){
      row[nz + idx] = I[ j*(nn+1) + idx + 1 ];
    }
    nz += nn;
  }
  col[n] = nz;

  if (nz != (nn*n) ) std::cerr << "Problem with kNN graph..." << std::endl;

  P.n   = n;
  P.m   = n;
  P.nnz = n * nn;
  P.row = row;
  P.col = col;
  P.val = val;
  
  return P;
  
}
