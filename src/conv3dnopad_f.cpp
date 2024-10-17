#include "3d_helper_f.cpp"

void conv3dnopad_f( double * const PhiGrid,
                    const double * const VGrid,
                    const double h,
                    uint32_t * const nGridDims,
                    const uint32_t nVec,
                    const uint32_t nDim,
                    const uint32_t np ) {

  struct timeval start;
  start = tsne_start_timer();

  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  fftwf_complex *K, *X, *w;
  std::complex<float> *Kc, *Xc, *wc;
  fftwf_plan planc_kernel, planc_rhs, planc_inverse;

  // get h^2
  float hsq = h*h;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1=nGridDims[0];
  uint32_t n2=nGridDims[1];
  uint32_t n3=nGridDims[2];

  int rank = 3;
  int n[] = {static_cast<int>(n1), static_cast<int>(n2), static_cast<int>(n3)};
  int howmany = nVec;
  int idist = n1*n2*n3;
  int odist = n1*n2*n3;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;



  // allocate memory for kernel and RHS FFTs
  K = (fftwf_complex *) fftwf_malloc( n1 * n2 * n3 * sizeof(fftwf_complex) );
  X = (fftwf_complex *) fftwf_malloc( n1 * n2 * n3 * nVec * sizeof(fftwf_complex) );
  w = (fftwf_complex *) fftwf_malloc( n1 * sizeof(fftwf_complex) );

  Kc = reinterpret_cast<std::complex<float> *> (K);
  Xc = reinterpret_cast<std::complex<float> *> (X);
  wc = reinterpret_cast<std::complex<float> *> (w);

  // get twiddle factors
  CILK_FOR (int i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );

  CILK_FOR (long int i = 0; i < n1*n2*n3; i++)
    Kc[i] = 0.0;
  CILK_FOR (long int i = 0; i < n1*n2*n3*nVec; i++)
    Xc[i] = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM


  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS

  tsne_stop_timer("init", start); start = tsne_start_timer();

  planc_kernel = fftwf_plan_dft_3d(n1, n2, n3, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

  planc_rhs = fftwf_plan_many_dft(rank, n, howmany, X, inembed,
                                 istride, idist,
                                 X, onembed,
                                 ostride, odist,
                                 FFTW_FORWARD, FFTW_ESTIMATE);

  planc_inverse = fftwf_plan_many_dft(rank, n, howmany, X, inembed,
                                     istride, idist,
                                     X, onembed,
                                     ostride, odist,
                                     FFTW_BACKWARD, FFTW_ESTIMATE);

  tsne_stop_timer("plan", start); start = tsne_start_timer();

  // ============================== 8 KERNELS

  eee( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("eee", start); start = tsne_start_timer();

  oee( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("oee", start); start = tsne_start_timer();

  eoe( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("eoe", start); start = tsne_start_timer();

  ooe( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("ooe", start); start = tsne_start_timer();

  eeo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("eeo", start); start = tsne_start_timer();

  oeo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("oeo", start); start = tsne_start_timer();

  eoo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("eoo", start); start = tsne_start_timer();

  ooo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  tsne_stop_timer("ooo", start); start = tsne_start_timer();

  for (int iVec=0; iVec<nVec; iVec++){
    for (int k=0; k<n3; k++){
      for (int j=0; j<n2; j++){
        for (int i=0; i<n1; i++){
          PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *= 0.125 / ((float) n1*n2*n3);
        }
      }
    }
  }

  tsne_stop_timer("phi", start); start = tsne_start_timer();

  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftwf_destroy_plan( planc_kernel );
  fftwf_destroy_plan( planc_rhs );
  fftwf_destroy_plan( planc_inverse );

  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftwf_free( K );
  fftwf_free( X );
  fftwf_free( w );

  tsne_stop_timer("destroy", start); start = tsne_start_timer();

}
