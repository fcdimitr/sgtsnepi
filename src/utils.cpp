/*!
  \file   utils.cpp
  \brief  Auxilliary utilities.

  \author Dimitris Floros
  \date   2019-06-20
*/

#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>

#include <cilk/cilk_api.h>

void printParams(tsneparams P){

  std::cout << "Number of vertices: " << P.n << std::endl
            << "Embedding dimensions: " << P.d << std::endl
            << "Rescaling parameter λ: " << P.lambda << std::endl
            << "Early exag. multiplier α: " << P.alpha << std::endl
            << "Maximum iterations: " << P.maxIter << std::endl
            << "Early exag. iterations: " << P.earlyIter << std::endl
            << "Learning rate: " << P.eta << std::endl;

  if (P.run_exact)
    std::cout << "Running exact QQ (quadratic complexity)" << std::endl;
  else
    if (P.h[0] >= P.maxIter)
      std::cout << "Box side length h: " << P.h[1] << std::endl;
    else
      std::cout << "Adaptive box side length h (changing with iterations) " << std::endl;

  std::cout << "Drop edges originating from leaf nodes? " << P.dropLeaf << std::endl
            << "Number of processes: " << P.np << std::endl;
  
}

double randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}

sparse_matrix buildPFromMTX( const char *filename ){

  // ~~~~~~~~~~ variable declarations
  sparse_matrix P;
  matval *val_coo;
  matidx *row_coo, *col_coo;

  // ~~~~~~~~~~ read matrix
  
  // open the file
  std::ifstream fin( filename );

  // ignore headers and comments
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> P.m >> P.n >> P.nnz;

  // allocate space for COO format
  val_coo = new matval [P.nnz];
  row_coo = new matidx [P.nnz];
  col_coo = new matidx [P.nnz];

  // read the COO data
  for (int l = 0; l < P.nnz; l++)
    fin >> row_coo[l] >> col_coo[l] >> val_coo[l];

  // close connection to file
  fin.close();

  // ~~~~~~~~~~ transform COO to CSC
  P.val = new matval [P.nnz];
  P.row = new matidx [P.nnz];
  P.col = new matidx [P.n+1] ();

  // ----- find the correct column sizes
  for (int l = 0; l < P.nnz; l++){            
    P.col[ col_coo[l]-1 ]++;
  }

  for(int i = 0, cumsum = 0; i < P.n; i++){     
    int temp = P.col[i];
    P.col[i] = cumsum;
    cumsum += temp;
  }
  P.col[P.n] = P.nnz;
  
  // ----- copy the row indices to the correct place
  for (int l = 0; l < P.nnz; l++){
    int col = col_coo[l]-1;
    int dst = P.col[col];
    P.row[dst] = row_coo[l]-1;
    P.val[dst] = val_coo[l];
    
    P.col[ col ]++;
  }
  
  // ----- revert the column pointers
  for(int i = 0, last = 0; i < P.n; i++) {     
    int temp = P.col[i];
    P.col[i] = last;

    last = temp;
  }

  // ~~~~~~~~~~ deallocate memory
  delete [] val_coo;
  delete [] row_coo;
  delete [] col_coo;

  // ~~~~~~~~~~ return value
  return P;
  
}


void extractEmbeddingText( double *y, int n, int d ){

  std::ofstream f ("embedding.txt");

  if (f.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          f << y[i*d + j] << " ";
        }
        f << std::endl;
      }

      f.close();
    }   
  
}


void extractEmbedding( double *y, int n, int d ){

  FILE * pFile;
  pFile = fopen ("embedding.bin", "wb");

  fwrite(y , sizeof(double), n*d, pFile);
  fclose(pFile);
  return;
}

int getWorkers(){
  return __cilkrts_get_nworkers();
}

void setWorkers(int n){
  std::string str = std::to_string(n);

  __cilkrts_end_cilk();
  if ( 0!=__cilkrts_set_param("nworkers", str.c_str() ) )
    std::cerr << "Error setting workers" << std::endl;
}

double * readXfromMTX( const char *filename, int *n, int *d ){

  // ~~~~~~~~~~ variable declarations
  double *X;
  // ~~~~~~~~~~ read matrix
  
  // open the file
  std::ifstream fin( filename );

  // ignore headers and comments
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> n[0] >> d[0];

  // allocate space for COO format
  X = new double [n[0] * d[0]];

  // read the COO data
  for (int j = 0; j < d[0]; j++)
    for (int l = 0; l < n[0]; l++)
      fin >> X[l*d[0] + j];

  // close connection to file
  fin.close();


  // ~~~~~~~~~~ return value
  return X;
  
}

// ##################################################
// FUNCTIONS IMPLEMENTED BY VAN DER MAATEN TO INPUT/OUTPUT DATA
// IN THE SAME FORMAT

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool vdm_load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

  // Open file, read first 2 integers, allocate memory, and read the data
  FILE *h;
  if((h = fopen("data.dat", "r+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return false;
  }
  fread(n, sizeof(int), 1, h);             // number of datapoints
  fread(d, sizeof(int), 1, h);             // original dimensionality
  fread(theta, sizeof(double), 1, h);      // gradient accuracy
  fread(perplexity, sizeof(double), 1, h); // perplexity
  fread(no_dims, sizeof(int), 1, h);       // output dimensionality
  fread(max_iter, sizeof(int),1,h);        // maximum number of iterations
  *data = new double [*d * *n];
  if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  fread(*data, sizeof(double), *n * *d, h);         // the data
  if(!feof(h)) fread(rand_seed, sizeof(int), 1, h); // random seed
  fclose(h);
  printf("Read the %i x %i data matrix successfully!\n", *n, *d);
  return true;
}

// Function that saves map to a t-SNE file
void vdm_save_data(double* data, int* landmarks, double* costs, int n, int d) {

  // Open file, write first 2 integers and then the data
  FILE *h;
  if((h = fopen("result.dat", "w+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return;
  }
  fwrite(&n, sizeof(int), 1, h);
  fwrite(&d, sizeof(int), 1, h);
  fwrite(data, sizeof(double), n * d, h);
  fwrite(landmarks, sizeof(int), n, h);
  fwrite(costs, sizeof(double), n, h);
  fclose(h);
  printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
