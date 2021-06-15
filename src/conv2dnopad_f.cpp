void conv2dnopad_f( double * const PhiGrid,
                    const double * const VGrid,
                    const double h,
                    uint32_t * const nGridDims,
                    const uint32_t nVec,
                    const uint32_t nDim,
                    const uint32_t np ) {


  printf("Single precision FFTW\n");

  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  fftwf_complex *K, *X, *w;
  std::complex<float> *Kc, *Xc, *wc;
  fftwf_plan planc_kernel, planc_rhs, planc_inverse;

  // get h^2
  float hsq = h*h;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1=nGridDims[0];
  uint32_t n2=nGridDims[1];

  int rank = 2;
  int n[] = {static_cast<int>(n1), static_cast<int>(n2)};
  int howmany = nVec;
  int idist = n1*n2;
  int odist = n1*n2;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;



  // allocate memory for kernel and RHS FFTs
  K = (fftwf_complex *) fftwf_malloc( n1 * n2 * sizeof(fftwf_complex) );
  X = (fftwf_complex *) fftwf_malloc( n1 * n2 * nVec * sizeof(fftwf_complex) );
  w = (fftwf_complex *) fftwf_malloc( n1 * sizeof(fftwf_complex) );

  Kc = reinterpret_cast<std::complex<float> *> (K);
  Xc = reinterpret_cast<std::complex<float> *> (X);
  wc = reinterpret_cast<std::complex<float> *> (w);

  // get twiddle factors
  for (int i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );

  for (long int i = 0; i < n1*n2; i++)
    Kc[i] = 0.0;
  for (long int i = 0; i < n1*n2*nVec; i++)
    Xc[i] = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM


  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS
  struct timeval start = tsne_start_timer();

  planc_kernel = fftwf_plan_dft_2d(n1, n2, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

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

  // ============================== EVEN-EVEN

  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<float> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      += tmp;
      if (i>0) Kc[SUB2IND2D(n1-i,j,n1)] += tmp;
      if (j>0) Kc[SUB2IND2D(i,n2-j,n1)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND2D(n1-i,n2-j,n1)] += tmp;
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ];
      }
    }
  }


  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);

  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftwf_execute(planc_inverse);

  // ---------- (no conjugate multiplication)

  for (int iVec=0; iVec<nVec; iVec++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] =
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }

  // ============================== ODD-EVEN


  for (long int i = 0; i < n1*n2; i++)
    Kc[i] = 0.0;
  for (long int i = 0; i < n1*n2*nVec; i++)
    Xc[i] = 0.0;
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<float> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      += tmp;
      if (i>0) Kc[SUB2IND2D(n1-i,j,n1)] -= tmp;
      if (j>0) Kc[SUB2IND2D(i,n2-j,n1)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND2D(n1-i,n2-j,n1)] -= tmp;
    }
  }


  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[i];
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          ( (float) VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] ) * wc[i];
      }
    }
  }


  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);

  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftwf_execute(planc_inverse);

  // ---------- data normalization
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[i]);
      }
    }
  }

  for (int iVec=0; iVec<nVec; iVec++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }


  // ============================== EVEN-ODD


  for (long int i = 0; i < n1*n2; i++)
    Kc[i] = 0.0;
  for (long int i = 0; i < n1*n2*nVec; i++)
    Xc[i] = 0.0;
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<float> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      += tmp;
      if (i>0) Kc[SUB2IND2D(n1-i,j,n1)] += tmp;
      if (j>0) Kc[SUB2IND2D(i,n2-j,n1)] -= tmp;
      if (i>0 && j>0) Kc[SUB2IND2D(n1-i,n2-j,n1)] -= tmp;
    }
  }


  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[j];
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          ( (float) VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] ) * wc[j];
      }
    }
  }


  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);

  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftwf_execute(planc_inverse);

  // ---------- data normalization
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[j]);
      }
    }
  }

  for (int iVec=0; iVec<nVec; iVec++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }


  // ============================== ODD-ODD


  for (long int i = 0; i < n1*n2; i++)
    Kc[i] = 0.0;
  for (long int i = 0; i < n1*n2*nVec; i++)
    Xc[i] = 0.0;
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<float> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      += tmp;
      if (i>0) Kc[SUB2IND2D(n1-i,j,n1)] -= tmp;
      if (j>0) Kc[SUB2IND2D(i,n2-j,n1)] -= tmp;
      if (i>0 && j>0) Kc[SUB2IND2D(n1-i,n2-j,n1)] += tmp;
    }
  }
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[j]*wc[i];
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          ( (float) VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] ) * wc[j] * wc[i];
      }
    }
  }


  // ---------- execute kernel plan
  fftwf_execute(planc_kernel);

  // ---------- execute RHS plan
  fftwf_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (int jVec=0; jVec<nVec; jVec++) {
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftwf_execute(planc_inverse);

  // ---------- data normalization
  for (int iVec=0; iVec<nVec; iVec++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[i]) * std::conj(wc[j]);
      }
    }
  }

  for (int iVec=0; iVec<nVec; iVec++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }

  for (int iVec=0; iVec<nVec; iVec++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] *= 0.25 / ((float) n1*n2);
      }
    }
  }


  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftwf_destroy_plan( planc_kernel );
  fftwf_destroy_plan( planc_rhs );
  fftwf_destroy_plan( planc_inverse );

  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftwf_free( K );
  fftwf_free( X );
  fftwf_free( w );

}
