/*!
  \file   demo_perplexity_equalization.cpp
  \brief  Conventional t-SNE usage.

  \author Dimitris Floros
  \date   2019-06-24
*/


#include <iostream>
#include <string>
#include <unistd.h>
#include <cmath>
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
  p.trees = 8;
  // p.log_level = FLANN_LOG_INFO;
  p.checks = 100;
  
  // -------- Run a kNN search
  flann_find_nearest_neighbors_double(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}


int main(int argc, char **argv)
{
  // ~~~~~~~~~~ variable declarations
  int opt;
  double u = 30;
  
  tsneparams params;
  std::string filename = "test.mtx";
  coord *y;

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') ) {
    // there is NO input...
    std::cerr << "No filename provided!" << std::endl;
    return 1;
  }
  else {
    // there is an input...
    filename = argv[argc-1];
  }

  // ----- retrieve optional arguments

  // Shut GetOpt error messages down (return '?'): 
  opterr = 0;

  while ( (opt = getopt(argc, argv, "d:a:m:e:h:p:u:")) != -1 ) { 
    switch ( opt ) {
    case 'd':
      params.d = atoi(optarg);
      break;
    case 'm':
      params.maxIter = atoi(optarg);
      break;
    case 'e':
      params.earlyIter = atoi(optarg);
      break;
    case 'p':
      params.np = atoi(optarg);
      break;
    case 'u':
      sscanf(optarg, "%lf", &u);
      break;
    case 'a':
      sscanf(optarg, "%lf", &params.alpha);
      break;
    case 'h':
      sscanf(optarg, "%lf", &params.h);
      break;
    case '?':  // unknown option...
      std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
      break;
    }
  }

  int nn = std::ceil( u*3 );

  params.lambda = 1;

  // ~~~~~~~~~~ get number of active workers
  params.np = getWorkers();

  // ~~~~~~~~~~ read input data points
  int n, d;
  double * X = readXfromMTX( filename.c_str(), &n, &d );

  params.n = n;
  
  // ~~~~~~~~~~ run kNN search

  std::cout << "Running k-neareast neighbor search for " << nn << " neighbors..."
            << std::flush;

  double * D = new double [params.n * (nn+1)];
  int    * I = new int    [params.n * (nn+1)];

  allKNNsearch(I, D, X, n, d, nn+1);

  std::cout << "DONE" << std::endl;
  
  sparse_matrix P = perplexityEqualization( I, D, n, nn, u );

  delete [] D;
  delete [] I;

  params.n = n;

  // ~~~~~~~~~~ Run SG-t-SNE
  y = sgtsne( P, params );

  // ~~~~~~~~~~ export results
  extractEmbedding( y, params.n, params.d );

  delete [] y;

}
