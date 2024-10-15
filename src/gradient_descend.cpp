/*!
  \file   gradient_descend.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include "gradient_descend.hpp"
#include <cmath>
#include <cfloat>
#include <cstdlib>

#include "cilk.hpp"

#include "sgtsne.hpp"
#include "timers.hpp"
#include "opadd_reducer.hpp"
#include "pq.hpp"
#include "qq.hpp"

template <class dataPoint>
void compute_dy(dataPoint       * const dy,
		dataPoint const * const Fattr,
		dataPoint const * const Frep,
		int               const N,
		int               const dim,
                dataPoint         const alpha){


  CILK_FOR (int k = 0; k < N*dim; k++)
    dy[k] = alpha * Fattr[k] - Frep[k];

}

template <class dataPoint>
void update_positions(dataPoint * const dY,
		      dataPoint * const uY,
		      int         const N,
		      int         const no_dims,
		      dataPoint * const Y,
		      dataPoint * const gains,
		      double      const momentum,
		      double      const eta){


  // Update gains
  CILK_FOR(int i = 0; i < N * no_dims; i++){
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if(gains[i] < .01) gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }

  // find mean
  dataPoint meany[no_dims];
  for (int i = 0; i < no_dims; i++){
    opadd_reducer<dataPoint> sum = 0.0;
    CILK_FOR (int j = i; j < N*no_dims; j += no_dims)
      sum += Y[j];
    meany[i] = static_cast<dataPoint>(sum) / N;
  }

  // zero-mean
  CILK_FOR(int n = 0; n < N; n++) {
    for(int d = 0; d < no_dims; d++) {
      Y[n*no_dims + d] -= meany[d];
    }
  }
  
}


template <class dataPoint>
double compute_gradient(dataPoint *dy,
                        double *timeFrep,
                        double *timeFattr,
			tsneparams params,
			dataPoint *y,
      sparse_matrix *P,
                        double *timeInfo){


  // ----- parse input parameters
  int d = params.d;
  int n = params.n;

  // ----- timing
  struct timeval start;

  // ----- Allocate memory
  dataPoint * Fattr = new dataPoint [n*d] ();
  dataPoint * Frep  = new dataPoint [n*d] ();

  // ------ Compute PQ (fattr)
  start = tsne_start_timer();
  pq(Fattr, y, P->val, P->row, P->col, n, d);
  if (timeInfo != nullptr) {
    timeInfo[0] = tsne_stop_timer("PQ", start);
    *timeFattr += timeInfo[0];
  } else
    *timeFattr += tsne_stop_timer("PQ", start);
  
  // ------ Compute QQ (frep)
  start = tsne_start_timer();
  double zeta;
  if (timeInfo != nullptr)
    zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np,
                                           &timeInfo[1]);
  else
    zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np);
  *timeFrep += tsne_stop_timer("QQ", start);
  // double zeta = computeFrepulsive_exact(Frep, y, n, d);

  // ----- Compute gradient (dY)
  compute_dy(dy, Fattr, Frep, n, d, params.alpha);
  
  // ----- Free-up memory
  delete [] Fattr;
  delete [] Frep;
  return zeta;
}


// Evaluate t-SNE cost function
double tsne_cost(
    matidx * row_P,
    matidx * col_P,
    matval * val_P,
    double * y,
    double zeta,
    int n,
    int d){

  // Adapted original cost function from:
  // github.com/lvdmaaten/bhtsne/blob/cd619e6c186b909a2d8ed26fbf0b1afec770f43d/tsne.cpp

  // Loop over all edges to compute t-SNE error
  int index_i, index_j;
  double C = 0.0, Q;
  for(int i = 0; i < n; i++) {
      index_i = i * d;
      for(int j = col_P[i]; j < col_P[i + 1]; j++) {
          Q = .0;
          index_j = row_P[j] * d;
          for(int k = 0; k < d; k++) {
            double y_diff  = y[index_i + k] - y[index_j + k];
            Q += y_diff * y_diff;
          }
          Q = (1.0 / (1.0 + Q)) / zeta;
          C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
      }
  }

  return C;
}

void kl_minimization(coord* y,
                     tsneparams params, 
                     sparse_matrix *P,
                     double **timeInfo = nullptr){

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int    stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta    = 200.0;
  int    iterPrint = 50;
  
  double timeFattr = 0.0;
  double timeFrep  = 0.0;

  struct timeval start;

  int    n = params.n;
  int    d = params.d;
  int    max_iter = params.maxIter;

  coord zeta = 0;

  // ----- Allocate memory
  coord* dy    = new coord [n*d];
  coord* uy    = new coord [n*d];
  coord* gains = new coord [n*d];

  // ------ Initialize
  for(int i = 0; i < n*d; i++){
    uy[i] =  .0;
    gains[i] = 1.0;
  }  

  // ----- Print precision
  if (sizeof(y[0]) == 4)
    std::cout << "Working with single precision" << std::endl;
  else if (sizeof(y[0]) == 8)
    std::cout << "Working with double precision" << std::endl;

  // ----- Start t-SNE iterations
  start = tsne_start_timer();
  for(int iter = 0; iter < max_iter; iter++) {

    // ----- Gradient calculation
    if (timeInfo == nullptr)
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, P);
    else
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, P,
                              timeInfo[iter]);
    // ----- Position update
    update_positions<coord>(dy, uy, n, d, y, gains, momentum, eta);

    // Stop lying about the P-values after a while, and switch momentum
    if(iter == stop_lying_iter) {
      params.alpha = 1;
    }

    // Change momentum after a while
    if(iter == mom_switch_iter){
      momentum = final_momentum;
    }
    
    // Print out progress
    if( iter % iterPrint == 0 || iter == max_iter - 1 ) {
      matval C = tsne_cost(P->row, P->col, P->val, y, zeta, n, d);
      if(iter == 0){
        std::cout << "Iteration " << iter+1
                  << ": error is " << C
                  << std::endl;
	
      } else {
        double iterTime = tsne_stop_timer("QQ", start);
        std::cout << "Iteration " << iter
                  << ": error is " << C
                  << " (50 iterations in " << iterTime
                  << " seconds)"
                  << std::endl;

        start = tsne_start_timer();
      }
    }
    
  }

  // ----- Print statistics (time spent at PQ and QQ)
  std::cout << " --- Time spent in each module --- \n" << std::endl;
  std::cout << " Attractive forces: " << timeFattr
            << " sec [" << timeFattr / (timeFattr + timeFrep) * 100
            << "%] |  Repulsive forces: " << timeFrep
            << " sec [" << timeFrep / (timeFattr + timeFrep) * 100
            << "%]" << std::endl;


  delete [] dy;
  delete [] uy;
  delete [] gains;
}


// ***** EXPLICIT INSTATIATION

template
double compute_gradient(double *dy,
                        double *timeFrep,
                        double *timeFattr,
			tsneparams params,
			double *y,
      sparse_matrix *P,
                        double *timeInfo);

// template
// double compute_gradient(float *dy,
//                         double *timeFrep,
//                         double *timeFattr,
// 			tsneparams params,
// 			float *y,
//      sparse_matrix *P,
//                         double *timeInfo);
