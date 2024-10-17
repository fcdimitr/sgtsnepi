void conv1dnopad_f( double * const PhiGrid,
                    const double * const VGrid,
                    const double h,
                    uint32_t * const nGridDims,
                    const uint32_t nVec,
                    const uint32_t nDim,
                    const uint32_t np) {

  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  fftwf_complex *K, *X, *w;
  std::complex<float> *Kc, *Xc, *wc;
  fftwf_plan planc_kernel, planc_rhs, planc_inverse;

  // get h^2
  float hsq = h*h;

  // total number of grid points (VGrid's leading dimension)
  uint32_t n1=nGridDims[0];

  // FFTW plan options
  int rank = 1;
  int n[] = {static_cast<int>(n1)};
  int howmany = nVec;
  int idist = n1;
  int odist = n1;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;



  // allocate memory for kernel and RHS FFTs
  K = (fftwf_complex *) fftwf_malloc( n1 * sizeof(fftwf_complex) );
  X = (fftwf_complex *) fftwf_malloc( n1 * nVec * sizeof(fftwf_complex) );
  w = (fftwf_complex *) fftwf_malloc( n1 * sizeof(fftwf_complex) );

  Kc = reinterpret_cast<std::complex<float> *> (K);
  Xc = reinterpret_cast<std::complex<float> *> (X);
  wc = reinterpret_cast<std::complex<float> *> (w);

  // get twiddle factors
  CILK_FOR (int i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );


  CILK_FOR (long int i = 0; i < n1; i++)
    Kc[i] = 0.0;
  CILK_FOR (long int i = 0; i < n1*nVec; i++)
    Xc[i] = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM



  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS

  struct timeval start = tsne_start_timer();
  planc_kernel = fftwf_plan_dft_1d(n1, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

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
  tsne_stop_timer( "PLAN", start );

  // ============================== EVEN FREQUENCIES

  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

  for (int i=0; i<n1; i++) {
    std::complex<float> tmp( kernel1d( hsq, i ), 0 );
             Kc[i]    += tmp;
    if (i>0) Kc[n1-i] += tmp;
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

  for (int iVec=0; iVec<nVec; iVec++) {
    for (int i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec ,n1) ] =
        VGrid[ SUB2IND2D(i, iVec, n1) ];
    }
  }

  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);


  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int i=0; i<n1; i++){
      Xc[SUB2IND2D(i,jVec,n1)] = Xc[SUB2IND2D(i,jVec,n1)] *
        Kc[i];
    }
  }

  // ---------- execute inverse plan
  fftwf_execute(planc_inverse);

  // ---------- (no conjugate multiplication)

  for (int iVec=0; iVec<nVec; iVec++){
    for (int i=0; i<n1; i++){
      PhiGrid[ SUB2IND2D(i, iVec, n1) ] =
        Xc[ SUB2IND2D(i, iVec, n1) ].real();
    }
  }

  // ============================== ODD FREQUENCIES


  CILK_FOR (long int i = 0; i < n1; i++)
    Kc[i] = 0.0;
  CILK_FOR (long int i = 0; i < n1*nVec; i++)
    Xc[i] = 0.0;
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (int i=0; i<n1; i++) {
    std::complex<float> tmp( kernel1d( hsq, i ), 0 );
             Kc[i]    += tmp;
    if (i>0) Kc[n1-i] -= tmp;
  }

  for (int i=0; i<n1; i++) {
    Kc[i] *= wc[i];
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

  for (int iVec=0; iVec<nVec; iVec++) {
    for (int i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec ,n1) ] =
        ( (float) VGrid[ SUB2IND2D(i, iVec, n1) ] ) * wc[i];
    }
  }

  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);

  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int i=0; i<n1; i++){
      Xc[SUB2IND2D(i,jVec,n1)] = Xc[SUB2IND2D(i,jVec,n1)] *
        Kc[i];
    }
  }

  // ---------- execute inverse plan
  fftwf_execute(planc_inverse);


  // ---------- data normalization
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec, n1) ] =
        Xc[ SUB2IND2D(i, iVec, n1) ] *
        std::conj(wc[i]);
    }
  }

  for (int iVec=0; iVec<nVec; iVec++){
    for (int i=0; i<n1; i++){
      PhiGrid[ SUB2IND2D(i, iVec, n1) ] +=
        Xc[ SUB2IND2D(i, iVec, n1) ].real();
    }
  }

  CILK_FOR (long int i = 0; i < n1*nVec; i++)
    PhiGrid[i] *= (0.5 / n1);

  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftwf_destroy_plan( planc_kernel );
  fftwf_destroy_plan( planc_rhs );
  fftwf_destroy_plan( planc_inverse );

  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftwf_free( K );
  fftwf_free( X );
  fftwf_free( w );

}
