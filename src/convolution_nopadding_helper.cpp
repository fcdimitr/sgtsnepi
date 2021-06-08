/*!
  \file   convolution_nopadding_helper.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-04-30
*/


#ifndef _CONVOLUTION_NOPADDING_HELPER_H_
#define _CONVOLUTION_NOPADDING_HELPER_H_

#include <cilk/cilk.h>
#include <cilk/cilkscale.h>

extern wsp_t __CS_NUCONV_KERNEL_ZERO;
extern wsp_t __CS_NUCONV_KERNEL_SETUP;
extern wsp_t __CS_NUCONV_KERNEL_FFTW_EXEC;
extern wsp_t __CS_NUCONV_KERNEL_HADAMARD;
extern wsp_t __CS_NUCONV_KERNEL_POSTPROC;


void eee( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    // REVIEW Is `cilk_for` better than `std::memset` for initializing arrays to 0?
    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (long k=0; k<n3; k++) {
    for (long j=0; j<n2; j++) {
        for (long i=0; i<n1; i++) {
            std::complex<double> tmp( kernel3d( hsq, i, j, k ), 0 );
            Kc[SUB2IND3D(   i,   j,   k, n1,n2)] += tmp;
            Kc[SUB2IND3D(n1-i,   j,   k, n1,n2)] += (i>0) ? tmp : 0;
            Kc[SUB2IND3D(   i,n2-j,   k, n1,n2)] += (j>0) ? tmp : 0;
            Kc[SUB2IND3D(n1-i,n2-j,   k, n1,n2)] += (i>0 && j>0) ? tmp : 0;
            Kc[SUB2IND3D(   i,   j,n3-k, n1,n2)] += (k>0) ? tmp : 0;
            Kc[SUB2IND3D(n1-i,   j,n3-k, n1,n2)] += (k>0 && i>0) ? tmp : 0;
            Kc[SUB2IND3D(   i,n2-j,n3-k, n1,n2)] += (k>0 && j>0) ? tmp : 0;
            Kc[SUB2IND3D(n1-i,n2-j,n3-k, n1,n2)] += (k>0 && i>0 && j>0) ? tmp : 0;
        }
    }
}
  
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

cilk_for (long i = 0; i < n1*n2*n3*nVec; i++)
    Xc[i] = VGrid[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- (no conjugate multiplication)

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3*nVec; i++)
    PhiGrid[i] = Xc[i].real();

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}


void oee( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[i];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

cilk_for (long jj = 0; jj < n2*n3*nVec; jj++)
    for (int i = 0; i < n1; i++)
        Xc[jj*n1 + i] = VGrid[jj*n1 + i] * wc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

cilk_for (long jj = 0; jj < n2*n3*nVec; jj++) {
    for (int i = 0; i < n1; i++) {
        Xc[     jj*n1 + i] *= std::conj(wc[i]);
        PhiGrid[jj*n1 + i] += Xc[jj*n1 + i].real();
    }
}

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}

void eoe( double * const PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);


// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int32_t iVec=0; iVec<nVec; iVec++) {
  for (int32_t k=0; k<n3; k++) {
    for (int32_t j=0; j<n2; j++) {
      for (int32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);


// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *= std::conj(wc[j]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}

void ooe( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[i];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (uint32_t iVec=0; iVec<nVec; iVec++) {
  for (uint32_t k=0; k<n3; k++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[i];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *=
           std::conj(wc[j]) * std::conj(wc[i]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}


void eeo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();
  
    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int32_t iVec=0; iVec<nVec; iVec++) {
  for (int32_t k=0; k<n3; k++) {
    for (int32_t j=0; j<n2; j++) {
      for (int32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[k];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *= std::conj(wc[k]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}

void oeo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[i] * wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int32_t iVec=0; iVec<nVec; iVec++) {
  for (int32_t k=0; k<n3; k++) {
    for (int32_t j=0; j<n2; j++) {
      for (int32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[i] * wc[k];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *=
           std::conj(wc[i]) * std::conj(wc[k]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}

void eoo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[k];
     }
   }
 }
 
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int32_t iVec=0; iVec<nVec; iVec++) {
  for (int32_t k=0; k<n3; k++) {
    for (int32_t j=0; j<n2; j++) {
      for (int32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[k];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);

// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *=
           std::conj(wc[j]) * std::conj(wc[k]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
           Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}

void ooo( double *PhiGrid, const double *VGrid,
          std::complex<double> *Xc, std::complex<double> *Kc, std::complex<double> *wc,
          fftw_plan planc_kernel, fftw_plan planc_rhs, fftw_plan planc_inverse,
          uint32_t n1, uint32_t n2, uint32_t n3, uint32_t nVec,
          double hsq ) {

    wsp_t tic = wsp_getworkspan();

    cilk_for (long int i = 0; i < n1*n2*n3; i++)
        Kc[i] = 0.0;
    cilk_for (long int i = 0; i < n1*n2*n3*nVec; i++)
        Xc[i] = 0.0;

    wsp_t toc = wsp_getworkspan();
    __CS_NUCONV_KERNEL_ZERO += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

tic = wsp_getworkspan();

for (int32_t k=0; k<n3; k++) {
  for (int32_t j=0; j<n2; j++) {
    for (int32_t i=0; i<n1; i++) {
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
 for (int32_t k=0; k<n3; k++) {
   for (int32_t j=0; j<n2; j++) {
     for (int32_t i=0; i<n1; i++) {
       Kc[SUB2IND3D(i,j,k,n1,n2)] *= wc[j] * wc[i] * wc[k];
     }
   }
 }
   
// ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
for (int32_t iVec=0; iVec<nVec; iVec++) {
  for (int32_t k=0; k<n3; k++) {
    for (int32_t j=0; j<n2; j++) {
      for (int32_t i=0; i<n1; i++) {
        Xc[ SUB2IND4D(i, j, k, iVec ,n1, n2, n3) ] =
          VGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] * wc[j] * wc[i] * wc[k];
      }
    }
  }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_SETUP += (toc - tic);


tic = wsp_getworkspan();

// ---------- execute kernel plan
fftw_execute(planc_kernel);
  
// ---------- execute RHS plan
fftw_execute(planc_rhs);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT

tic = wsp_getworkspan();

cilk_for (long i = 0; i < n1*n2*n3; i++)
    for (int jVec = 0; jVec < nVec; jVec++)
        Xc[jVec*n1*n2*n3 + i] *= Kc[i];

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_HADAMARD += (toc - tic);

// ---------- execute plan

tic = wsp_getworkspan();

fftw_execute(planc_inverse);

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_FFTW_EXEC += (toc - tic);

// ---------- data normalization

tic = wsp_getworkspan();

 for (int32_t iVec=0; iVec<nVec; iVec++) {
   for (int32_t k=0; k<n3; k++){
     for (int32_t j=0; j<n2; j++) {
       for (int32_t i=0; i<n1; i++) {
         Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *=
           std::conj(wc[j]) * std::conj(wc[i]) * std::conj(wc[k]);
         PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] +=
          Xc[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ].real();
       }
     }
   }
 }

toc = wsp_getworkspan();
__CS_NUCONV_KERNEL_POSTPROC += (toc - tic);

}


#endif /* _CONVOLUTION_NOPADDING_HELPER_H_ */
