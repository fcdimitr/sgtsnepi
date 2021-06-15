/*!
  \file   convolution_nopadding_helper.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-04-30
*/


#ifndef _CONVOLUTION_NOPADDING_HELPER_H_
#define _CONVOLUTION_NOPADDING_HELPER_H_

void eee( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {


    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (long k=0; k<n3; k++) {
  for (long j=0; j<n2; j++) {
    for (long i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] += tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] += tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] += tmp;
    }
  }
 }
  
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

cilk_for (long i = 0; i < n1*n2*n3*nVec; i++)
    Xc[i] = VGrid[i];

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (long jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- (no conjugate multiplication)

cilk_for (long i = 0; i < n1*n2*n3*nVec; i++)
    PhiGrid[i] = Xc[i].real();

}


void oee( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

  
// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] -= tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] -= tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] -= tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[i];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[i];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[i]);
       }
     }
   }
 }
  
 for (int iVec=0; iVec<nVec; iVec++){
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++){
       for (int i=0; i<n1; i++){
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

}

void eoe( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;


// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] += tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] -= tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] -= tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] -= tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[j]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] += 
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}

void ooe( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] -= tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] -= tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] += tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] += tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[i];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[i];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[j]) * std::conj(wc[i]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}


void eeo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

  
    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;
// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] += tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] += tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] -= tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[k];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[k]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}

void oeo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] -= tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] += tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] -= tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] += tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[i] * wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[i] * wc[k];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[i]) * std::conj(wc[k]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}

void eoo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)]                                 += tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)]                     -= tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)]                     += tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)]           -= tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)]                     -= tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)]           += tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)]           -= tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] += tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[k];
     }
   }
 }
 
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[k];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);

// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse); 

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[j]) * std::conj(wc[k]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}

void ooo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
for (int k=0; k<n3; k++) {
  for (int j=0; j<n2; j++) {
    for (int i=0; i<n1; i++) {
      std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
      Kc[SUB2IND3D(i,j,k,n1,n2)] += tmp;
      if (i>0) Kc[SUB2IND3D(n1-i,j,k,n1,n2)] -= tmp;
      if (j>0) Kc[SUB2IND3D(i,n2-j,k,n1,n2)] -= tmp;
      if (i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,k,n1,n2)] += tmp;
      if (k>0) Kc[SUB2IND3D(i,j,n3-k,n1,n2)] -= tmp;
      if (k>0 && i>0) Kc[SUB2IND3D(n1-i,j,n3-k,n1,n2)] += tmp;
      if (k>0 && j>0) Kc[SUB2IND3D(i,n2-j,n3-k,n1,n2)] += tmp;
      if (k>0 && i>0 && j>0) Kc[SUB2IND3D(n1-i,n2-j,n3-k,n1,n2)] -= tmp;
    }
  }
 }
 for (int k=0; k<n3; k++) {
   for (int j=0; j<n2; j++) {
     for (int i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[i] * wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int iVec=0; iVec<nVec; iVec++) {
  for (int k=0; k<n3; k++) {
    for (int j=0; j<n2; j++) {
      for (int i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[i] * wc[k];
      }
    }
  }
 }


// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
for (int jVec=0; jVec<nVec; jVec++) {
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] = Xc[SUB2IND4D(i,j,k,jVec,n1,n2,n3)] *
          Kc[SUB2IND3D(i,j,k,n1,n2)];
      }
    }
  }
 }

// ---------- execute plan
fftw_execute(planc_inverse);

// ---------- data normalization
 for (int iVec=0; iVec<nVec; iVec++) {
   for (int k=0; k<n3; k++){
     for (int j=0; j<n2; j++) {
       for (int i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] =
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *
           std::conj(wc[j]) * std::conj(wc[i]) * std::conj(wc[k]);
       }
     }
   }
 }
  
for (int iVec=0; iVec<nVec; iVec++){
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n1; i++){
        PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
      }
    }
  }
 }

}


#endif /* _CONVOLUTION_NOPADDING_HELPER_H_ */
