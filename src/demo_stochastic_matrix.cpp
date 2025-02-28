/*!
  \file   demo_stochastic_matrix.cpp
  \brief  Demo usage of SG-t-SNE-Pi 
  Demo usage of SG-t-SNE-Pi with a sparse stochastic graph stored in 
  Matrix Market format. The embedding [N-by-d] is exported in a binary 
  file in row-major storage.

  \author Dimitris Floros
  \date   2019-06-21
*/


#include <iostream>
#include <string>
#include <unistd.h>

#include "sgtsne.hpp"

int main(int argc, char **argv)
{
  // ~~~~~~~~~~ variable declarations
  int opt;
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

  while ( (opt = getopt(argc, argv, "l:d:a:m:e:h:p:r")) != -1 ) {
    switch ( opt ) {
    case 'l':
      sscanf(optarg, "%lf", &params.lambda);
      break;
    case 'd':
      params.d = atoi(optarg);
      break;
    case 'r':
      sscanf(optarg, "%lf", &params.eta);
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

  // ~~~~~~~~~~ setup number of workers

#ifndef OPENCILK
  if (getWorkers() != params.np && params.np > 0)
    setWorkers( params.np );
#endif

  params.np = getWorkers();

  // ~~~~~~~~~~ load stochastic graph
  sparse_matrix P = buildPFromMTX( filename.c_str() );
  params.n = P.m;

  // ~~~~~~~~~~ Run SG-t-SNE
  y = sgtsne( P, params );

  // ~~~~~~~~~~ export results
  extractEmbeddingText( y, params.n, params.d );

  // ~~~~~~~~~~ print instruction to show the graph
  std::cout << std::endl << " If gnuplot is installed, issue" << std::endl << std::endl
            << "   gnuplot -e 'set size square; plot \"embedding.txt\" with dots'" << std::endl
            << std::endl << " to visualize the embedding" << std::endl << std::endl;
  

  delete [] y;

}
