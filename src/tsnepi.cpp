/*!
  \file   tsnepi.cpp
  \brief  Support of the conventional t-SNE

  \author Dimitris Floros
  \date   2019-07-11
*/


#include <iostream>
#include <string>
#include <unistd.h>
#include <flann/flann.h>

#include "sgtsne.hpp"

//! Compute the approximate all-kNN graph of the input data points
/*!  
  Compute the k-nearest neighbor dataset points of every point in
  the datsaet set using approximate k-nearest neighbor search (FLANN).

*/
void allKNNsearch(int * IDX,        //!< [k-by-N] array with the neighbor IDs
                  double * DIST,    //!< [k-by-N] array with the neighbor distances
                  double * dataset, //!< [L-by-N] array with coordinates of data points
                  int N,            //!< [scalar] Number of data points N
                  int dims,         //!< [scalar] Number of dimensions L
                  int kappa) {      //!< [scalar] Number of neighbors k

  
  struct FLANNParameters p; 

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 16;
  p.checks = 300;
  
  // -------- Run a kNN search
  flann_find_nearest_neighbors_double(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}

int main()
{
  // ~~~~~~~~~~ variable declarations
  int origN, N, L, no_dims, max_iter;
  double perplexity, theta, *data;
  int rand_seed = -1;
  tsneparams params;
  coord *y;

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if (vdm_load_data(&data, &origN, &L, &no_dims, &theta, &perplexity, &rand_seed, &max_iter)) {

    N = origN;

    params.lambda = 1;
    params.maxIter = max_iter;
    params.d = no_dims;
    params.n = N;

    // using perplexity, get 3*u neighbors
    int nn = std::ceil( perplexity*3 );
    
    // get workers
    params.np = getWorkers();

    // ~~~~~~~~~~ run kNN search

    std::cout << "Running k-neareast neighbor search for " << nn << " neighbors..."
              << std::flush;

    double * D = new double [params.n * (nn+1)];
    int    * I = new int    [params.n * (nn+1)];

    allKNNsearch(I, D, data, N, L, nn+1);

    std::cout << "DONE" << std::endl;

    std::cout << "Perplexity equalization u = " << perplexity << "..."
              << std::flush;
    
    sparse_matrix P = perplexityEqualization( I, D, N, nn, perplexity );

    std::cout << "DONE" << std::endl;

    delete [] D;
    delete [] I;

    // ~~~~~~~~~~ Run SG-t-SNE
    y = sgtsne( P, params );

    int* landmarks = new int [N];
    for(int n = 0; n < N; n++) landmarks[n] = n;

    double* costs = new double [N] ();
    vdm_save_data(y, landmarks, costs, N, no_dims);

    // Clean up the memory
    delete [] data;
    delete [] y;
    delete [] costs;
    delete [] landmarks;
  }
  
  
}

