#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mmintrin.h>  //  MMX
#include <xmmintrin.h> //  SSE
#include <emmintrin.h> //  SSE 2  
#include <pmmintrin.h> //  SSE 3  

#ifndef AMD
	#include <tmmintrin.h> // SSSE 3  
	#include <smmintrin.h> //  SSE 4.1
	#include <nmmintrin.h> //  SSE 4.2
	#include <wmmintrin.h> //  SSE ?? (AES)
#else
	#include <ammintrin.h>   // SSE4A (amd's popcount)
	#include <x86intrin.h>
#endif

#include "timer.clock_gettime.c"


//---------------------------------------
// Type Definitions
//---------------------------------------

typedef   signed char      ssp_s8;
typedef unsigned char      ssp_u8;

typedef   signed short     ssp_s16;
typedef unsigned short     ssp_u16;

typedef   signed int       ssp_s32;
typedef unsigned int       ssp_u32;

typedef float              ssp_f32;
typedef double             ssp_f64;

typedef   signed long long ssp_s64;
typedef unsigned long long ssp_u64;

typedef union
{
__m128  f;
__m128d d;
__m128i i;
__m64       m64[ 2];
ssp_u64 u64[ 2];
ssp_s64 s64[ 2];
ssp_f64 f64[ 2];
ssp_u32 u32[ 4];
ssp_s32 s32[ 4];
ssp_f32 f32[ 4];
ssp_u16 u16[ 8];
ssp_s16 s16[ 8];
ssp_u8  u8 [16];
ssp_s8  s8 [16];
} ssp_m128;


/** \SSE4_1{SSE2,_mm_blendv_pd} */
inline __m128d ssp_blendv_pd_SSE2( __m128d a, __m128d b, __m128d mask )
{
    ssp_m128 A, B, Mask;
    A.d = a;
    B.d = b;
    Mask.d = mask;

// _MM_SHUFFLE(z,y,x,w) does not select anything, this macro just creates a mask
// expands to the following value: (z<<6) | (y<<4) | (x<<2) | w

    Mask.i = _mm_shuffle_epi32( Mask.i, _MM_SHUFFLE(3, 3, 1, 1) );
    Mask.i = _mm_srai_epi32   ( Mask.i, 31                      );

    B.i = _mm_and_si128( B.i, Mask.i );
    A.i = _mm_andnot_si128( Mask.i, A.i );
    A.i = _mm_or_si128( A.i, B.i );
    return A.d;
}


#ifdef AMD
	#define _mm_blendv_pd ssp_blendv_pd_SSE2
#endif 

/**
 * SpMV (usually used as a subroutine) using bitmasked register blocks
 * This version works only with doubles and 8x8 register blocks 
 * @param[in] nbr number of register blocks
 * @param[in] bot the local part of the bottom array, i.e. {lower row bits}.{higher row bits}
 **/
template <typename IT>
void SSEspmv(const double *V, const uint64_t *M, const IT *bot, const IT nrb, const double *X, double *Y){
  const double *_V = V-1;
  __m128d Y01QW = _mm_setzero_pd();
  __m128d Y23QW = _mm_setzero_pd();
  __m128d Y45QW = _mm_setzero_pd();
  __m128d Y67QW = _mm_setzero_pd();
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(IT i=0;i<nbr;++i){
    const uint64_t Ci = C[i];
    const uint64_t Zi = ~M[i]; // a 1 denotes a zero

#ifdef AMD
   const uint64_t Zil = Zi << 1; 
   __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zi<<1,1); // Z01[0][63] = Z[63]
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);
    //--------------------------------------------------------------------------
    const __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0x8000000000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xE000000000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xF800000000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFE00000000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFF80000000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFE0000000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFF8000000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFE000000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFF800000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFE00000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFF80000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFE0000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFF8000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFE000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFF800000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFE00000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X44QW = _mm_loaddup_pd(&X[4+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X44QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFF80000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X44QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFE0000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X44QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFF8000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X44QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFE000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X55QW = _mm_loaddup_pd(&X[5+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X55QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFF800000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X55QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFE00000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X55QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFF80000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X55QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFE0000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X66QW = _mm_loaddup_pd(&X[6+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X66QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFF8000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X66QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFE000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X66QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFF800)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X66QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFFE00)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    //--------------------------------------------------------------------------
    const __m128d X77QW = _mm_loaddup_pd(&X[7+Ci]);
    Y01QW = _mm_add_pd(_mm_mul_pd(X77QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFFF80)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);
    Y23QW = _mm_add_pd(_mm_mul_pd(X77QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFFFE0)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y23QW);
    Y45QW = _mm_add_pd(_mm_mul_pd(X77QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFFFF8)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y45QW);
    Y67QW = _mm_add_pd(_mm_mul_pd(X77QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[i]&0xFFFFFFFFFFFFFFFE)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);
    //--------------------------------------------------------------------------
    _V = (double*)((uint64_t)_V+__builtin_popcountll(M[i]));
    //--------------------------------------------------------------------------
  }
  _mm_store_pd(&Y[0],Y01QW);
  _mm_store_pd(&Y[2],Y23QW);
  _mm_store_pd(&Y[4],Y45QW);
  _mm_store_pd(&Y[6],Y67QW);
}

int main (){
  timer_init();
  uint64_t i,trial,trials=1000;

  uint64_t N = 10000;
    double *V = (  double*)malloc(N*64*sizeof(  double));for(i=0;i<N*64;i++)V[i]=1.0*(i+1);
  uint64_t *M = (uint64_t*)malloc(N* 1*sizeof(uint64_t));for(i=0;i<N* 1;i++)M[i]=0x0000000000000000;
  uint32_t *C = (uint32_t*)malloc(N* 1*sizeof(uint32_t));for(i=0;i<N* 1;i++)C[i]=0;
  uint32_t *P = (uint32_t*)malloc(N* 1*sizeof(uint32_t));for(i=0;i<N* 1;i++)P[i]=0;
  
  double *X = (  double*)malloc(1024*sizeof(  double));for(i=0;i<1024;i++)X[i]=1.0;
  double *Y = (  double*)malloc(1024*sizeof(  double));for(i=0;i<1024;i++)Y[i]=1.0;

  double t0 = timer_seconds_since_init();
  for(trial=0;trial<trials;trial++){
    SpMV(V,M,C,P,N,X,Y);
    double *temp=Y;Y=X;X=temp;
  }
  double t1 = timer_seconds_since_init();

  printf("%0.9f seconds\n",t1-t0);
  printf("%4.3f GFlop/s\n",128*N*trials/(t1-t0)/1e9);
  for(i=0;i<8;i++)printf("%6.3f ",X[i]);printf("\n");
  for(i=0;i<8;i++)printf("%6.3f ",Y[i]);printf("\n");
  return 0;
}
