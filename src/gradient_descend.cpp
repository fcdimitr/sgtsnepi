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
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <vector>

#include <fftw3.h>

#include "timers.hpp"
#include "qq.hpp"

#define MAX_NV_EXACT 2000  // maximum number of vertices to run exact

extern std::vector<double> GLOBAL_GRID_SIZES;

template <class dataPoint>
void compute_dy(dataPoint       * const dy,
		dataPoint const * const Fattr,
		dataPoint const * const Frep,
		int               const N,
		int               const dim,
                dataPoint         const alpha){


  cilk_for (int k = 0; k < N*dim; k++)
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
          double      const eta,
          double      const bound_box){


  // Update gains
  cilk_for(int i = 0; i < N * no_dims; i++){
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if(gains[i] < .01) gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }

  // find mean
  dataPoint meany[no_dims];
  for (int i = 0; i < no_dims; i++){
    cilk::reducer_opadd<dataPoint> meany_reducer(0.0);
    cilk_for (int j = i; j < N*no_dims; j += no_dims)
      *meany_reducer += Y[j];
    meany[i] = meany_reducer.get_value() / N;
  }

  // maximum across all dimensions
  double max_tot = 0.0;

  // zero-mean
  cilk_for(int n = 0; n < N; n++) {
    for(int d = 0; d < no_dims; d++) {
      Y[n*no_dims + d] -= meany[d];
      if( std::abs( Y[n*no_dims + d] ) > max_tot )
        max_tot = std::abs( Y[n*no_dims + d] );
    }
  }

  // shrink
  if ( max_tot > bound_box ) {
    max_tot /= bound_box;
    GLOBAL_GRID_SIZES.push_back( max_tot );
    cilk_for(int n = 0; n < N; n++) {
      for(int d = 0; d < no_dims; d++) {
        Y[n*no_dims + d] /= max_tot;
      }
    }
  } else
    GLOBAL_GRID_SIZES.push_back( 1.0 );

}


template <class dataPoint>
double compute_gradient(dataPoint *dy,
                        double *timeFrep,
                        double *timeFattr,
                        tsneparams params,
                        dataPoint *y,
                        BiCsb<dataPoint, unsigned int> * csb,
                        int p,
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
  csb_pq( NULL, NULL, csb, y, Fattr, n, d, 0, 0, 0 );
  if (timeInfo != nullptr) {
    timeInfo[0] = tsne_stop_timer("PQ", start);
    *timeFattr += timeInfo[0];
  } else
    *timeFattr += tsne_stop_timer("PQ", start);
  
  // ------ Compute QQ (frep)
  start = tsne_start_timer();
  double zeta;
  if (n <= MAX_NV_EXACT || params.run_exact){
    zeta = computeFrepulsive_exact(Frep, y, n, d);
    GLOBAL_GRID_SIZES.push_back( 0.0 );
    GLOBAL_GRID_SIZES.push_back( 0.0 );
  } else {
    if (timeInfo != nullptr)
      zeta = computeFrepulsive_interp(Frep, y, n, d, params.h[p+1], params.np, params.fftw_single,
                                      &timeInfo[1]);
    else
      zeta = computeFrepulsive_interp(Frep, y, n, d, params.h[p+1], params.np, params.fftw_single);
  }

  *timeFrep += tsne_stop_timer("QQ", start);
  // double zeta = computeFrepulsive_exact(Frep, y, n, d);

  // ----- Compute gradient (dY)
  compute_dy(dy, Fattr, Frep, n, d, params.alpha);
  
  // ----- Free-up memory
  delete [] Fattr;
  delete [] Frep;
  return zeta;
}

void kl_minimization(coord* y,
                     tsneparams params, 
                     BiCsb<matval, matidx> *csb,
                     double **timeInfo = nullptr){

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int    stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta    = params.eta;
  int    iterPrint = 50;
  
  double timeFattr = 0.0;
  double timeFrep  = 0.0;

  struct timeval start;

  int    n = params.n;
  int    d = params.d;
  int    max_iter = params.maxIter;

  coord zeta = 0;

  // ----- setup parallel FFTW
  fftw_init_threads();
  fftw_plan_with_nthreads( params.np );
  fftwf_init_threads();
  fftwf_plan_with_nthreads( params.np );

  if ( params.fftw_single )
    std::cout << "Setting-up parallel (single-precision) FFTW: " << params.np << std::endl;
  else
    std::cout << "Setting-up parallel (double-precision) FFTW: " << params.np << std::endl;

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
  // if (sizeof(y[0]) == 4)
  //   std::cout << "Working with single precision" << std::endl;
  // else if (sizeof(y[0]) == 8)
  //   std::cout << "Working with double precision" << std::endl;

  // phase index
  int p = 0;

  // ----- Start t-SNE iterations
  start = tsne_start_timer();
  for(int iter = 0; iter < max_iter; iter++) {

    // check phase transition
    if (iter == params.h[p]) p += 2;

    // ----- Gradient calculation
    if (timeInfo == nullptr)
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, csb, p);
    else
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, csb, p,
                              timeInfo[iter]);
    // ----- Position update
    update_positions<coord>(dy, uy, n, d, y, gains, momentum, eta, params.bound_box);

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
      matval C = tsne_cost( csb, y, n, d, params.alpha, zeta );
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
                        BiCsb<double, unsigned int> * csb,
                        int p,
                        double *timeInfo);

// template
// double compute_gradient(float *dy,
//                         double *timeFrep,
//                         double *timeFattr,
// 			tsneparams params,
// 			float *y,
// 			BiCsb<float, unsigned int> * csb,
//                         double *timeInfo);
