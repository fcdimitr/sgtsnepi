#include "utils.hpp"
#include "cs.hpp"
#include <string.h>

template <typename T>
void verifyVectorEqual(T const * const f_new, T const * const f_gold,
                       CS_INT n, CS_INT dim,
                       double const ERR_THRES){

  for (CS_INT i=0; i<n; i++) {

    bool isEqual = true;

    for (CS_INT j=0; j<dim; j++){
      isEqual &= fabs(f_new[i*dim + j] - f_gold[i*dim + j]) < ERR_THRES;
    }
    
    if (!isEqual) {
      printf("\nVALIDATION ERROR!\n\n");
      std::cout << "f_new[" << i << "]  = (";
      for (CS_INT j=0; j<dim-1; j++)
	printf("%.5g,", f_new[i*dim+j]);
      printf("%.5g)\n", f_new[i*dim+dim-1]);

      std::cout << "f_gold[" << i << "]  = (";
      for (CS_INT j=0; j<dim-1; j++)
	printf("%.5g,", f_gold[i*dim+j]);
      printf("%.5g)\n", f_gold[i*dim+dim-1]);


      printf("\n");
#ifdef CHECK_SOFT
      return;
#else
      exit(1);
#endif
    }
  }
  

}


cs * triu( cs const * const A ){

  // original matrix sizes
  CS_INT m       = A->m;
  CS_INT n       = A->n;
  CS_INT *Ap     = A->p;
  CS_INT *Ai     = A->i;
  double *Ax  = A->x;
  CS_INT nzmax   = A->nzmax;
  CS_INT nz      = A->nz;

  CS_INT *nElem = (CS_INT *) calloc( m, sizeof( CS_INT ) );
  CS_INT nznew  = 0;
  
  // --- COUNTING
  
  // loop through every column
  for (CS_INT j = 0 ; j < m ; j++) {

    // get range for NNZ for current col
    for (CS_INT p = Ap [j] ; p < Ap [j+1] ; p++) {

      // if element is in upper triangular keep
      if (j >= Ai[p]){
        nElem[j]++;
        nznew++;
      }
      
    }
    
  }

#ifdef DEBUG
  for (CS_INT j = 0; j < n; j++)
    printf("%d ", nElem[j]);
  printf("\n");
#endif
  
  // allocate new matrix
  cs *C_sym = cs_spalloc (m, n, nznew, 1, 0) ;

  CS_INT *Ap_sym    = C_sym->p;
  CS_INT *Ai_sym    = C_sym->i;
  double *Ax_sym = C_sym->x;

  // --- COPYING

  // first element is zero
  CS_INT offset = 0;

  for (CS_INT j=0; j<n; j++){

    // copy number of rows in each column
    Ap_sym[j] = offset;
    offset   += nElem[j];
    
  }

  // fill last element
  Ap_sym[n] = offset;

  // counter looping through all nnz indexes
  CS_INT k_sym = 0;
  
  // loop through every col
  for (CS_INT j = 0 ; j < m ; j++) {

    // get range for NNZ for current col
    for (CS_INT p = Ap [j] ; p < Ap [j+1] ; p++) {

      // if element is in upper triangular keep
      if (j >= Ai[p]){
        
        // copy index and value
        Ai_sym[ k_sym ] = Ai[p];
        Ax_sym[ k_sym ] = Ax[p];

        // increment counter
        k_sym++;
      }
      
    }
    
  }

#ifdef DEBUG
  printf( "NNZ: %d, k_sym: %d", nznew, k_sym );
#endif
  
  free( nElem );

  return C_sym;
  
}


void printDenseMatrix( double *matrix, CS_INT m, CS_INT n ) {

  for (CS_INT row=0; row<m; row++) {
    for(CS_INT columns=0; columns<n; columns++)
      printf("%.4f ", matrix[row + m*columns]);
    printf("\n");
  }
  
}

cs *band2sparse( double *B, CS_INT n, CS_INT b ) {

  cs *C;

  C = cs_spalloc (n, n, n*b, 1, 1) ;

  for ( CS_INT i = 0; i < n; i++ ) {
    for ( CS_INT j = 0; j < 2*b+1; j++ ) {

      CS_INT r,c;

      r = j - b + i;
      c = i;
      if ( r>=0 && r<n )
        cs_entry (C, r, c, B[j+i*(2*b+1)]);

    }

  }

  return C;
  
}

cs *band2sparseLim( double *B, CS_INT n, CS_INT b, CS_INT lim ) {

  cs *C;

  CS_INT count = 0;
  
  C = cs_spalloc (n, n, n*b, 1, 1) ;


  for ( CS_INT j = 0; j < b; j++ ) {
    for ( CS_INT i = 0; i < n; i++ ) {
      
      if (count == lim)
        return C;
      
      CS_INT r,c;

      r = j - b + i;
      c = i;
      if ( r>=0 && r<n ){
        cs_entry (C, r, c, B[j+i*(2*b+1)]);     // add element
        
        count++;
          
      }

    }

  }

  return C;
  
}

cs *genSymBandSparse( CS_INT n, CS_INT b, CS_INT lim ) {

  cs *C;

  CS_INT count = 0;
  
  C = cs_spalloc (n, n, lim, 1, 1) ;


  for ( CS_INT j = 0; j < b; j++ ) {
    for ( CS_INT i = 0; i < n; i++ ) {
      
      if (count >= lim)
        return C;
      
      CS_INT r,c;

      r = j + i;
      c = i;
      if ( r>=0 && r<n ){

        double elem = (double)rand()/RAND_MAX + 0.5;
        
        cs_entry (C, r, c, elem);     // add element
        count++;
        
        if (r!=c){
          cs_entry (C, c, r, elem);   // symmetric element
          count++;
        }
          
      }

    }

  }

  return C;
  
}

CS_INT dropdiag (CS_INT i, CS_INT j, double aij, void *other) { return (i != j) ;}

cs *make_sym (cs *A) {
  cs *AT, *C ;
  AT = cs_transpose (A, 1) ;          /* AT = A' */
  cs_fkeep (AT, &dropdiag, NULL) ;    /* drop diagonal entries from AT */
  C = cs_add (A, AT, 1, 1) ;          /* C = A+AT */
  cs_spfree (AT) ;
  return (C) ;
}

double *generateBanded( CS_INT n, CS_INT b ) {

  srand ( time ( NULL));
  
  double *B = (double *)malloc(n*(2*b+1)*sizeof(double));
  
  for ( CS_INT i = 0; i<n; i++ )
    for ( CS_INT j = 0; j<(2*b+1); j++ )
      B[j+i*(2*b+1)] = (double)rand()/RAND_MAX + 0.5;

  return B;

}

void printMinTime( double *x, CS_INT n ) {

  std::cout << __sec_reduce_min(x[0:n])*1000 << " ms" << std::endl;

}
  


void exportTime2csv( double *x, FILE *fp, CS_INT n ) {

  for (CS_INT i=0; i<n-1; i++)
    fprintf(fp,"%.3f,",x[i]);
  
  fprintf(fp,"%.3f\n",x[n-1]);

}

std::string getHostnameDateFilename() {
  char hostname[1024];
  gethostname(hostname, 1024);

  time_t rawtime;
  struct tm * timeinfo;

  char buffer[80];
  
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  strftime(buffer,sizeof(buffer),"%Y-%m-%d %H-%M-%S",timeinfo);
  
  std::string info(hostname);
  std::string datetime(buffer);

  info = info + " " + datetime;

  std::replace( info.begin(), info.end(), ' ', '_');
  std::replace( info.begin(), info.end(), '.', '-');
  
  info = info + ".csv";

  return info;
  
}

void exportBenchmarkResults( std::string prefix, double **times, char **names,
                             CS_INT nExp, CS_INT iter ) {
  
  std::string fileName = prefix + "_" + getHostnameDateFilename();
  FILE *fp = fopen( fileName.c_str(), "w" );
  for (CS_INT i=0; i<nExp; i++){
    fprintf(fp, names[i]);
    fprintf(fp, ",");
    exportTime2csv( times[i], fp, iter);
  }
  fclose(fp);
            
}

void exportBenchmarkResult( std::string prefix, double *times,
                            CS_INT iter ) {
  
  std::string fileName = prefix + "_" + getHostnameDateFilename();
  FILE *fp = fopen( fileName.c_str(), "w" );
  exportTime2csv( times, fp, iter);
  fclose(fp);
            
}



template<typename T>
T *permuteDataPoints( T* x, CS_INT *p, CS_INT n, CS_INT ldim ){

  T *y = (T *)malloc(n*ldim*sizeof(T));

  CS_INT i;
  
  cilk_for( i=0; i<n; i++ ){
    y[i*ldim:ldim] = x[p[i]*ldim:ldim];
  }

  return y;
  
}


void extractDimensions(double *y, double *x, CS_INT N, CS_INT ldim, CS_INT d){

  for(CS_INT i=0; i<N; i++){
    for(CS_INT j=0; j<d; j++){
      y[i*d + j] = x[i*ldim + j];
    }
  }

}

void setThreadsNum(int nworkers){

  char strw[10];
  sprintf(strw, "%d", nworkers);
  __cilkrts_end_cilk();
  __cilkrts_set_param("nworkers",strw);

}

CS_INT StartsWith(const char *a, const char *b)
{
   if(strncmp(a, b, strlen(b)) == 0) return 1;
   return 0;
}

// ==================================================
// EXPLICIT INSTANTIATIONS FOR LINKING

template
double *permuteDataPoints( double* x, CS_INT *p, CS_INT n, CS_INT ldim );

template
float *permuteDataPoints( float* x, CS_INT *p, CS_INT n, CS_INT ldim );


template
void verifyVectorEqual(float const * const f_new, float const * const f_gold,
                       CS_INT n, CS_INT dim, double const ERR_THRES);

template
void verifyVectorEqual(double const * const f_new, double const * const f_gold,
                       CS_INT n, CS_INT dim, double const ERR_THRES);
