#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <mmintrin.h>  //  MMX
#include <xmmintrin.h> //  SSE
#include <emmintrin.h> //  SSE 2  
#include <pmmintrin.h> //  SSE 3  

#ifndef AMD
	#include <tmmintrin.h> // SSSE 3  
	#include <smmintrin.h> //  SSE 4.1
	#include <nmmintrin.h> //  SSE 4.2
	#include <wmmintrin.h> //  SSE ?? (AES)
#endif

#ifdef ICC
	#include <nmmintrin.h>
#else
	#include <ammintrin.h>   // SSE4A (amd's popcount)
#endif

using namespace std;


const uint64_t masktable64[64] = {0x8000000000000000, 0x4000000000000000, 0x2000000000000000, 0x1000000000000000,
				0x0800000000000000, 0x0400000000000000, 0x0200000000000000, 0x0100000000000000,
				0x0080000000000000, 0x0040000000000000, 0x0020000000000000, 0x0010000000000000,
				0x0008000000000000, 0x0004000000000000, 0x0002000000000000, 0x0001000000000000,
				0x0000800000000000, 0x0000400000000000, 0x0000200000000000, 0x0000100000000000,
				0x0000080000000000, 0x0000040000000000, 0x0000020000000000, 0x0000010000000000,
				0x0000008000000000, 0x0000004000000000, 0x0000002000000000, 0x0000001000000000,
				0x0000000800000000, 0x0000000400000000, 0x0000000200000000, 0x0000000100000000,
				0x0000000080000000, 0x0000000040000000, 0x0000000020000000, 0x0000000010000000,
				0x0000000008000000, 0x0000000004000000, 0x0000000002000000, 0x0000000001000000,
				0x0000000000800000, 0x0000000000400000, 0x0000000000200000, 0x0000000000100000,
				0x0000000000080000, 0x0000000000040000, 0x0000000000020000, 0x0000000000010000,
				0x0000000000008000, 0x0000000000004000, 0x0000000000002000, 0x0000000000001000,
				0x0000000000000800, 0x0000000000000400, 0x0000000000000200, 0x0000000000000100,
				0x0000000000000080, 0x0000000000000040, 0x0000000000000020, 0x0000000000000010,
				0x0000000000000008, 0x0000000000000004, 0x0000000000000002, 0x0000000000000001 };


const unsigned short masktable16[16] = {0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 
					0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001 };

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


/** 
 * \SSE4_1{SSE2,_mm_blendv_pd} 
 * ISSUE: Do not short-circuit, i.e. loads 'a' regardless of the mask value
 * Question: Does the original blendv_pd (in SSE4.1) short-circuit?
 */
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

#ifdef ICC
	#define __builtin_popcountll _mm_popcnt_u64
	#define __builtin_popcount _mm_popcnt_u32
#endif 

// 16-bit reversal table
const unsigned char BitReverseTable64[] =
{
 0x0, 0x20, 0x10, 0x30, 0x8, 0x28, 0x18, 0x38,                  
 0x4, 0x24, 0x14, 0x34, 0xc, 0x2c, 0x1c, 0x3c,                  
 0x2, 0x22, 0x12, 0x32, 0xa, 0x2a, 0x1a, 0x3a,                   
 0x6, 0x26, 0x16, 0x36, 0xe, 0x2e, 0x1e, 0x3e,                   
 0x1, 0x21, 0x11, 0x31, 0x9, 0x29, 0x19, 0x39,                   
 0x5, 0x25, 0x15, 0x35, 0xd, 0x2d, 0x1d, 0x3d,                   
 0x3, 0x23, 0x13, 0x33, 0xb, 0x2b, 0x1b, 0x3b,                   
 0x7, 0x27, 0x17, 0x37, 0xf, 0x2f, 0x1f, 0x3f
};


// reverse 16-bit value, 6 bits at time
unsigned short BitReverse(unsigned short v)
{
	unsigned short c = (BitReverseTable64[v & 0x3f] << 10) | 
    	(BitReverseTable64[(v >> 6) & 0x3f] << 4) | 
    	(BitReverseTable64[(v >> 12) & 0x0f] >> 2);

        return c;
}


inline void atomicallyIncrementDouble(volatile double *target, const double by){
  asm volatile(                                            
    "movq  %0, %%rax \n\t"			// rax = *(%0)
    "xorpd %%xmm0, %%xmm0 \n\t"			// xmm0 = [0.0,0.0]
    "movsd %1, %%xmm0\n\t"			// xmm0[lo] = *(%1)
    "1:\n\t"                 
    // rax (containing *target) was last set at startup or by a failed cmpxchg
    "movq  %%rax,  %%xmm1\n\t"			// xmm1[lo] = rax
    "addsd %%xmm0, %%xmm1\n\t"			// xmm1 = xmm0 + xmm1 = by + xmm1
    "movq  %%xmm1, %%r8  \n\t"			// r8 = xmm1[lo]
    "lock cmpxchgq %%r8, %0\n\t"		// if(*(%0)==rax){ZF=1;*(%0)=r8}else{ZF=0;rax=*(%0);}
    "jnz 1b\n\t"                                // jump back if failed (ZF=0)
    : "=m"(*target)                             // outputs
    : "m"(by)     		                // inputs
    : "cc", "memory", "%rax", "%r8", "%xmm0", "%xmm1" // clobbered
  );
  return;
}

void symcsr(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlowbits)
{
	static const size_t NMortonRows64[] =
	{ 
 		0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
 		4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
 		0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
 		4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7
	};
	static const size_t NMortonCols64[] =
	{ 
 		0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3,
 		0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3,
 		4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7,
 		4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7
	};

	for(unsigned i=0; i<nrb;++i)
	{
    		const unsigned Ci = bot[i] & lowmask;
    		const unsigned Ri = (bot[i] >> nlowbits) & lowmask;
		uint64_t mask = M[i];
		for(size_t j=0; j<64; ++j)
		{
			if(mask & masktable64[j])
			{	atomicallyIncrementDouble(&Y[Ri+NMortonRows64[j]], (*V) * X[Ci+NMortonCols64[j]]);
				atomicallyIncrementDouble(&YT[Ci+NMortonCols64[j]], (*V) * XT[Ri+NMortonRows64[j]]);
				++V;
			}
		}
	}
}


void symcsr(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlowbits)
{
	static const size_t NMortonRows16[] = { 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3 };
	static const size_t NMortonCols16[] = { 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3 };

	for(unsigned i=0; i<nrb;++i)
	{
    		const unsigned Ci = bot[i] & lowmask;
    		const unsigned Ri = (bot[i] >> nlowbits) & lowmask;
		unsigned short mask = M[i];
		for(size_t j=0; j<16; ++j)
		{
			if(mask & masktable16[j])
			{	atomicallyIncrementDouble(&Y[Ri+NMortonRows16[j]], (*V) * X[Ci+NMortonCols16[j]]);
				atomicallyIncrementDouble(&YT[Ci+NMortonCols16[j]], (*V) * XT[Ri+NMortonRows16[j]]);
				++V;
			}
		}
	}
}

void symcsr(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlowbits)
{
	for(unsigned i=0; i<nrb;++i)
	{
    		const unsigned Ci = bot[i] & lowmask;
    		const unsigned Ri = (bot[i] >> nlowbits) & lowmask;
		unsigned char mask = M[i];
		if(mask & 0x8)
		{
			atomicallyIncrementDouble(&Y[Ri+0], (*V) * X[Ci+0]);
			atomicallyIncrementDouble(&YT[Ci+0], (*V) * XT[Ri+0]);
			V++;
		}

		if(mask & 0x4)
		{
			atomicallyIncrementDouble(&Y[Ri+1], (*V) * X[Ci+0]);
			atomicallyIncrementDouble(&YT[Ci+0], (*V) * XT[Ri+1]);
			V++;
		}
		
		if(mask & 0x2)
		{
			atomicallyIncrementDouble(&Y[Ri+0], (*V) * X[Ci+1]);
			atomicallyIncrementDouble(&YT[Ci+1], (*V) * XT[Ri+0]);
			V++;
		}
		
		if(mask & 0x1)
		{
			atomicallyIncrementDouble(&Y[Ri+1], (*V) * X[Ci+1]);
			atomicallyIncrementDouble(&YT[Ci+1], (*V) * XT[Ri+1]);
			V++;
		}
	}
}


/**
 * Symmetric SpMV inner kernel using bitmasked register blocks
 * 2-by-2 potentially diagonal case (X == XT and Y == YT) 
 * We can still use the __restrict keyword because we only use one alias for both X and XT
 **/
void SSEsym(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, double * __restrict Y, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 60); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X01QW = _mm_loadu_pd(&X[Ri]);	// the transpose of X aliases X itself

    //  {0,2}      02  {0,1} 
    //  {1,3}  <-  13  {2,3}

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8)])), _mm_setzero_pd(),(__m128d)Z01QW); // ERROR here ! [invalid read of _V, debug with sym matrix]
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE)])), _mm_setzero_pd(),(__m128d)Z23QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,A01QW),Y01QW);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,A23QW),Y01QW);
    __m128d Y00QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y11QW = _mm_mul_pd(X01QW, A23QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind] & 0x0F);  
    //--------------------------------------------------------------------------

    ssp_m128 yt0, yt1;
    yt0.d = Y00QW;
    yt1.d = Y11QW;
    
    _mm_store_pd(&Y[Ri],Y01QW);
    
    // The additional Y_T updates should come after we stored Y[Ri] back, otherwise they will be lost
    Y[Ci+0] += yt0.f64[0] + yt0.f64[1];
    Y[Ci+1] += yt1.f64[0] + yt1.f64[1];
  }
} 


/**
 * Symmetric SpMV inner kernel using bitmasked register blocks
 * 2-by-2 general case (X != XT and Y != YT) 
 * assumes strict-aliasing on X and Y
 **/
void SSEsym(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 60); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X01QW = _mm_loadu_pd(&XT[Ri]);	// use the transpose of X

    //  {0,2}      02  {0,1} 
    //  {1,3}  <-  13  {2,3}

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8)])), _mm_setzero_pd(),(__m128d)Z01QW);
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE)])), _mm_setzero_pd(),(__m128d)Z23QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,A01QW),Y01QW);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,A23QW),Y01QW);
    __m128d YT0QW = _mm_mul_pd(X01QW, A01QW);
    __m128d YT1QW = _mm_mul_pd(X01QW, A23QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind] & 0x0F);  
    //--------------------------------------------------------------------------

    ssp_m128 yt0, yt1;
    yt0.d = YT0QW;
    yt1.d = YT1QW;
   
    YT[Ci+0] += yt0.f64[0] + yt0.f64[1];
    YT[Ci+1] += yt1.f64[0] + yt1.f64[1];
    _mm_store_pd(&Y[Ri],Y01QW);
  }
} 


/**
 * SpMV (usually used as a subroutine) using bitmasked register blocks
 * This version works only with double values, unsigned indices, and 2x2 register blocks 
 * @param[in] nbr number of register blocks for this compressed sparse block only
 * @param[in] bot the local part of the bottom array, i.e. {lower row bits}.{higher row bits}
 * \attention 	SSEspmv should only be called within a single compressed sparse block and 
 * 		X and Y should already be partially indexed by the higher order bits
 * We don't need any template specialization based on the register block size
 * because for different block sizes, M's type differs, hence creating overloaded definitions
 **/
void SSEspmv(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lcmask;
    const unsigned Ri = (bot[ind] >> clbits) & lrmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 60); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);

    //  {0,2}      02  {0,1} 
    //  {1,3}  <-  13  {2,3}

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind] & 0x0F);  
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
  }
}

// 8x8 version, using uint64_t for M
// Possibly aliasing (Y=YT or X=XT) version for the blocks right on the diagonal
void SSEsym(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
		const double * __restrict X, double * Y, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;
    const uint64_t Zi = ~M[ind]; // a 1 denotes a zero
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);
    __m128d Y45QW = _mm_loadu_pd(&Y[Ri+4]);
    __m128d Y67QW = _mm_loadu_pd(&Y[Ri+6]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    __m128d X01QW = _mm_loadu_pd(&X[Ri]); // the transpose of X aliases X itself
    __m128d X23QW = _mm_loadu_pd(&X[Ri+2]); 
    __m128d X45QW = _mm_loadu_pd(&X[Ri+4]); 
    __m128d X67QW = _mm_loadu_pd(&X[Ri+6]); 

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0x8000000000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xE000000000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    __m128d A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xF800000000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    __m128d A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFE00000000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d Y00QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y11QW = _mm_mul_pd(X01QW, A23QW);
    Y00QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y00QW);
    Y11QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y11QW);

    // reuse variables for the second half of the first quadrand
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFF80000000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFE0000000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFF8000000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFE000000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d Y22QW = _mm_mul_pd(X01QW, A01QW);	// the transpose (lower-triangular) updates
    __m128d Y33QW = _mm_mul_pd(X01QW, A23QW);
    Y22QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y22QW);
    Y33QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y33QW);

    // Second quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFF800000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFE00000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFF80000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFE0000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    Y00QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), Y00QW);	// the transpose updates
    Y11QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), Y11QW);
    Y00QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), Y00QW);
    Y11QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), Y11QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFF8000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFE000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFF800000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFE00000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    Y22QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), Y22QW);	// the transpose updates
    Y33QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), Y33QW);
    Y22QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), Y22QW);
    Y33QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), Y33QW);


    // Reuse registers (e.g., X00QW <- X44QW)
    X00QW = _mm_loaddup_pd(&X[4+Ci]);
    X11QW = _mm_loaddup_pd(&X[5+Ci]);
    X22QW = _mm_loaddup_pd(&X[6+Ci]);
    X33QW = _mm_loaddup_pd(&X[7+Ci]);

    // Third quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFF80000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFE0000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFF8000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFE000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d Y44QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y55QW = _mm_mul_pd(X01QW, A23QW);
    Y44QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y44QW);
    Y55QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y55QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFF800000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFE00000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFF80000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFE0000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d Y66QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y77QW = _mm_mul_pd(X01QW, A23QW);
    Y66QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y66QW);
    Y77QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y77QW);

    // Fourth quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFF8000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFE000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFF800)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFE00)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    Y44QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), Y44QW);	// the transpose updates
    Y55QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), Y55QW);
    Y44QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), Y44QW);
    Y55QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), Y55QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFF80)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFE0)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFF8)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFFE)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y45QW); 	// no need to shift ZxxQW
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y67QW); 
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y45QW); 
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y67QW); 

    Y66QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), Y66QW);	// the transpose updates
    Y77QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), Y77QW);
    Y66QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), Y66QW);
    Y77QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), Y77QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcountll(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);
    _mm_store_pd(&Y[Ri+4],Y45QW);
    _mm_store_pd(&Y[Ri+6],Y67QW);

    // These mirror updates come after the stores, otherwise we lose the updates

    ssp_m128 yt0, yt1, yt2, yt3,yt4,yt5,yt6,yt7;
    yt0.d = Y00QW;
    yt1.d = Y11QW;
    yt2.d = Y22QW;
    yt3.d = Y33QW;
    yt4.d = Y44QW;
    yt5.d = Y55QW;
    yt6.d = Y66QW;
    yt7.d = Y77QW;
    
    Y[Ci+0] += yt0.f64[0] + yt0.f64[1];
    Y[Ci+1] += yt1.f64[0] + yt1.f64[1];
    Y[Ci+2] += yt2.f64[0] + yt2.f64[1];
    Y[Ci+3] += yt3.f64[0] + yt3.f64[1];
    Y[Ci+4] += yt4.f64[0] + yt4.f64[1];
    Y[Ci+5] += yt5.f64[0] + yt5.f64[1];
    Y[Ci+6] += yt6.f64[0] + yt6.f64[1];
    Y[Ci+7] += yt7.f64[0] + yt7.f64[1];
  }
}


// 8x8 version, using uint64_t for M
// No aliasing between Y and YT 
void SSEsym(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
		const double * __restrict X, const double * __restrict XT, double * __restrict Y, double * __restrict YT, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;
    const uint64_t Zi = ~M[ind]; // a 1 denotes a zero
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);
    __m128d Y45QW = _mm_loadu_pd(&Y[Ri+4]);
    __m128d Y67QW = _mm_loadu_pd(&Y[Ri+6]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    __m128d X01QW = _mm_loadu_pd(&XT[Ri]); 
    __m128d X23QW = _mm_loadu_pd(&XT[Ri+2]); 
    __m128d X45QW = _mm_loadu_pd(&XT[Ri+4]); 
    __m128d X67QW = _mm_loadu_pd(&XT[Ri+6]); 

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0x8000000000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xE000000000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    __m128d A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xF800000000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    __m128d A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFE00000000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d YT0QW = _mm_mul_pd(X01QW, A01QW);
    __m128d YT1QW = _mm_mul_pd(X01QW, A23QW);
    YT0QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT0QW);
    YT1QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT1QW);

    // reuse variables for the second half of the first quadrand
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFF80000000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFE0000000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFF8000000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFE000000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d YT2QW = _mm_mul_pd(X01QW, A01QW);	// the transpose (lower-triangular) updates
    __m128d YT3QW = _mm_mul_pd(X01QW, A23QW);
    YT2QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT2QW);
    YT3QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT3QW);

    // Second quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFF800000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFE00000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFF80000000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFE0000000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    YT0QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), YT0QW);	// the transpose updates
    YT1QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), YT1QW);
    YT0QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), YT0QW);
    YT1QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), YT1QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFF8000000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFE000000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFF800000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFE00000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    YT2QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), YT2QW);	// the transpose updates
    YT3QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), YT3QW);
    YT2QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), YT2QW);
    YT3QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), YT3QW);

    ssp_m128 yt0, yt1, yt2, yt3;
    yt0.d = YT0QW;
    yt1.d = YT1QW;
    yt2.d = YT2QW;
    yt3.d = YT3QW;

    YT[Ci+0] += yt0.f64[0] + yt0.f64[1];
    YT[Ci+1] += yt1.f64[0] + yt1.f64[1];
    YT[Ci+2] += yt2.f64[0] + yt2.f64[1];
    YT[Ci+3] += yt3.f64[0] + yt3.f64[1];

    // Reuse registers (e.g., X00QW <- X44QW)
    X00QW = _mm_loaddup_pd(&X[4+Ci]);
    X11QW = _mm_loaddup_pd(&X[5+Ci]);
    X22QW = _mm_loaddup_pd(&X[6+Ci]);
    X33QW = _mm_loaddup_pd(&X[7+Ci]);

    // Third quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFF80000000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFE0000000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFF8000000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFE000000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    YT0QW = _mm_mul_pd(X01QW, A01QW);	// reuse Y(1:4) registers for Y(5:8)
    YT1QW = _mm_mul_pd(X01QW, A23QW);
    YT0QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT0QW);
    YT1QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT1QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFF800000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFE00000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFF80000)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFE0000)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    YT2QW = _mm_mul_pd(X01QW, A01QW);
    YT3QW = _mm_mul_pd(X01QW, A23QW);
    YT2QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT2QW);
    YT3QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT3QW);

    // Fourth quadrand 
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFF8000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFE000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFF800)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFE00)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y45QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y67QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y45QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y67QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    YT0QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), YT0QW);	// the transpose updates
    YT1QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), YT1QW);
    YT0QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), YT0QW);
    YT1QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), YT1QW);

    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFF80)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFE0)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFF8)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFFE)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y45QW); 	// no need to shift ZxxQW
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y67QW); 
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y45QW); 
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y67QW); 

    YT2QW = _mm_add_pd(_mm_mul_pd(X45QW, A01QW), YT2QW);	// the transpose updates
    YT3QW = _mm_add_pd(_mm_mul_pd(X45QW, A23QW), YT3QW);
    YT2QW = _mm_add_pd(_mm_mul_pd(X67QW, A45QW), YT2QW);
    YT3QW = _mm_add_pd(_mm_mul_pd(X67QW, A67QW), YT3QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcountll(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);
    _mm_store_pd(&Y[Ri+4],Y45QW);
    _mm_store_pd(&Y[Ri+6],Y67QW);

    yt0.d = YT0QW;
    yt1.d = YT1QW;
    yt2.d = YT2QW;
    yt3.d = YT3QW;

    YT[Ci+4] += yt0.f64[0] + yt0.f64[1];
    YT[Ci+5] += yt1.f64[0] + yt1.f64[1];
    YT[Ci+6] += yt2.f64[0] + yt2.f64[1];
    YT[Ci+7] += yt3.f64[0] + yt3.f64[1];
  }
}



// Possibly aliasing (Y=YT or X=XT) version for the blocks right on the diagonal
void SSEsym(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
		const double * __restrict X, double * Y, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 48); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    __m128d X01QW = _mm_loadu_pd(&X[Ri]); // the transpose of X aliases X itself
    __m128d X23QW = _mm_loadu_pd(&X[Ri+2]); 

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    __m128d A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xF800)])), _mm_setzero_pd(),(__m128d)Z45QW);
    __m128d A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFE00)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d Y00QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y11QW = _mm_mul_pd(X01QW, A23QW);
    Y00QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y00QW);
    Y11QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y11QW);

    // reuse variables for the second half of A
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFF80)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFE0)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFF8)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFFE)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); // the shifts on ZxxQW are unnecessary after this point
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); 
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); 
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); 

    __m128d Y22QW = _mm_mul_pd(X01QW, A01QW);
    __m128d Y33QW = _mm_mul_pd(X01QW, A23QW);
    Y22QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), Y22QW);
    Y33QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), Y33QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);

    // These mirror updates come after the stores, otherwise we lose the updates
    ssp_m128 yt0, yt1, yt2, yt3;
    yt0.d = Y00QW;
    yt1.d = Y11QW;
    yt2.d = Y22QW;
    yt3.d = Y33QW;

    Y[Ci+0] += yt0.f64[0] + yt0.f64[1];
    Y[Ci+1] += yt1.f64[0] + yt1.f64[1];
    Y[Ci+2] += yt2.f64[0] + yt2.f64[1];
    Y[Ci+3] += yt3.f64[0] + yt3.f64[1];
  }
}

void SSEsym(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
		const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lowmask;
    const unsigned Ri = (bot[ind] >> nlbits) & lowmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 48); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    __m128d X01QW = _mm_loadu_pd(&XT[Ri]); 
    __m128d X23QW = _mm_loadu_pd(&XT[Ri+2]); 

    __m128d A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8000)])), _mm_setzero_pd(),(__m128d)Z01QW);
    __m128d A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE000)])), _mm_setzero_pd(),(__m128d)Z23QW);
    __m128d A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xF800)])), _mm_setzero_pd(),(__m128d)Z45QW);
    __m128d A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFE00)])), _mm_setzero_pd(),(__m128d)Z67QW);

    // Operations rescheduled for maximum parallelism (they follow a 1-3-2-4 order)
    //  {0,2}      02**  {0,1} 
    //  {1,3}  <-  13**  {2,3}
    //  *          ****  *
    //  *          ****  *

    //  *          ****  {4,5} 
    //  *      <-  ****  {6,7}
    //  {4,6}      46**  *
    //  {5,7}      57**  *
    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW, A01QW), Y01QW); Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW, A45QW), Y23QW); Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW, A23QW), Y01QW); Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW, A67QW), Y23QW); Z67QW=_mm_slli_epi64(Z67QW,8);

    __m128d YT0QW = _mm_mul_pd(X01QW, A01QW);
    __m128d YT1QW = _mm_mul_pd(X01QW, A23QW);
    YT0QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT0QW);
    YT1QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT1QW);

    // write YT back (Safe since we know that Y is not an alias to YT)
    ssp_m128 yt0, yt1;
    yt0.d = YT0QW;
    yt1.d = YT1QW;

    YT[Ci+0] += yt0.f64[0] + yt0.f64[1];
    YT[Ci+1] += yt1.f64[0] + yt1.f64[1];


    // reuse variables for the second half of A
    A01QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFF80)])), _mm_setzero_pd(),(__m128d)Z01QW);
    A23QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFE0)])), _mm_setzero_pd(),(__m128d)Z23QW);
    A45QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFF8)])), _mm_setzero_pd(),(__m128d)Z45QW);
    A67QW = _mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFFE)])), _mm_setzero_pd(),(__m128d)Z67QW);

    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW, A01QW), Y01QW); // the shifts on ZxxQW are unnecessary after this point
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW, A45QW), Y23QW); 
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW, A23QW), Y01QW); 
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW, A67QW), Y23QW); 

    __m128d YT2QW = _mm_mul_pd(X01QW, A01QW);
    __m128d YT3QW = _mm_mul_pd(X01QW, A23QW);
    YT2QW = _mm_add_pd(_mm_mul_pd(X23QW, A45QW), YT2QW);
    YT3QW = _mm_add_pd(_mm_mul_pd(X23QW, A67QW), YT3QW);

    // write YT back (Safe since we know that Y is not an alias to YT)
    ssp_m128 yt2, yt3;
    yt2.d = YT2QW;
    yt3.d = YT3QW;

    YT[Ci+2] += yt2.f64[0] + yt2.f64[1];
    YT[Ci+3] += yt3.f64[0] + yt3.f64[1];


    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);
  }
}


/**
 * SpMV (usually used as a subroutine) using bitmasked register blocks
 * This version works only with double values, unsigned indices, and 4x4 register blocks 
 * @param[in] nbr number of register blocks for this compressed sparse block only
 * @param[in] bot the local part of the bottom array, i.e. {lower row bits}.{higher row bits}
 * \attention 	SSEspmv should only be called within a single compressed sparse block and 
 * 		X and Y should already be partially indexed by the higher order bits
 * We don't need any template specialization based on the register block size
 * because for different block sizes, M's type differs, hence creating overloaded definitions
 **/
void SSEspmv(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lcmask;
    const unsigned Ri = (bot[ind] >> clbits) & lrmask;

    const uint64_t m64 = (uint64_t) M[ind]; // upcast to 64 bit, fill-in zeros from left 
    const uint64_t Zi = ((~m64) << 48); // a 1 denotes a zero 
    const uint64_t Zil = Zi << 1; 

#ifdef AMD
    __m128i Z01QW = _mm_unpacklo_epi64 (_mm_loadl_epi64((__m128i*)&Zi), _mm_loadl_epi64((__m128i*)&Zil));
#else
    __m128i Z01QW = _mm_insert_epi64(_mm_loadl_epi64((__m128i*)&Zi),Zil,1); 
#endif
    __m128i Z23QW = _mm_slli_epi64(Z01QW, 2);
    __m128i Z45QW = _mm_slli_epi64(Z01QW, 4);
    __m128i Z67QW = _mm_slli_epi64(Z01QW, 6);

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);

    //--------------------------------------------------------------------------
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    // Operations rescheduled for maximum parallelism (they follow a 1-3-2-4 order)
    //  {0,2}      02**  {0,1} 
    //  {1,3}  <-  13**  {2,3}
    //  *          ****  *
    //  *          ****  *

    //  *          ****  {4,5} 
    //  *      <-  ****  {6,7}
    //  {4,6}      46**  *
    //  {5,7}      57**  *
    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0x8000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xF800)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xE000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFE00)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    
    //  {8,0}      **80  *
    //  {9,1}  <-  **91  *
    //  *          ****  {8,9}
    //  *          ****  {0,1}

    //  *          ****  *
    //  *      <-  ****  *
    //  {2,4}      **24  {2,3}
    //  {3,5}      **35  {4,5}
    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFF80)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFF8)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFE0)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcount(M[ind]&0xFFFE)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcount(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);
  }
}

    //--------------------------------------------------------------------------
// M is of type uint64_t --> 8x8 register blocks
void SSEspmv(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits)
{
  const double * __restrict _V = V-1;
 
  // use popcnt to index into nonzero stream
  // use blendv where 1 = zero
  for(unsigned ind=0;ind<nrb;++ind)
  {
    const unsigned Ci = bot[ind] & lcmask;
    const unsigned Ri = (bot[ind] >> clbits) & lrmask;
    const uint64_t Zi = ~M[ind]; // a 1 denotes a zero

    __m128d Y01QW = _mm_loadu_pd(&Y[Ri]);
    __m128d Y23QW = _mm_loadu_pd(&Y[Ri+2]);
    __m128d Y45QW = _mm_loadu_pd(&Y[Ri+4]);
    __m128d Y67QW = _mm_loadu_pd(&Y[Ri+6]);

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
    __m128d X00QW = _mm_loaddup_pd(&X[0+Ci]);   // load and duplicate a double into 128-bit registers.
    __m128d X11QW = _mm_loaddup_pd(&X[1+Ci]);
    __m128d X22QW = _mm_loaddup_pd(&X[2+Ci]);
    __m128d X33QW = _mm_loaddup_pd(&X[3+Ci]);

    // Operations rescheduled for maximum parallelism (they follow a 1-3-2-4 order)
    //  {0,2}      02**  {0,1} 
    //  {1,3}  <-  13**  {2,3}
    //  *          ****  *
    //  *          ****  *

    //  *          ****  {4,5} 
    //  *      <-  ****  {6,7}
    //  {4,6}      46**  *
    //  {5,7}      57**  *
    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0x8000000000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xF800000000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xE000000000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFE00000000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    
    //  {8,0}      **80  *
    //  {9,1}  <-  **91  *
    //  *          ****  {8,9}
    //  *          ****  {0,1}

    //  *          ****  *
    //  *      <-  ****  *
    //  {2,4}      **24  {2,3}
    //  {3,5}      **35  {4,5}
    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFF80000000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFF8000000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFE0000000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFE000000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);Z67QW=_mm_slli_epi64(Z67QW,8);

    //--------------------------------------------------------------------------
    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFF800000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y45QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFF80000000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y67QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFE00000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y45QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFE0000000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFF8000000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y45QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFE000000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y45QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFF800000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y67QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFE00000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // Reuse registers (e.g., X00QW <- X44QW)
    X00QW = _mm_loaddup_pd(&X[4+Ci]);
    X11QW = _mm_loaddup_pd(&X[5+Ci]);
    X22QW = _mm_loaddup_pd(&X[6+Ci]);
    X33QW = _mm_loaddup_pd(&X[7+Ci]);

    Y01QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFF80000000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFF8000000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFE0000000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFE000000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    
    Y01QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFF800000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y01QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFF80000)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y23QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y01QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFE00000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y01QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y23QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFE0000)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y23QW);Z67QW=_mm_slli_epi64(Z67QW,8);
    
    Y45QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFF8000)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y45QW);Z01QW=_mm_slli_epi64(Z01QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X00QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFF800)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y67QW);Z45QW=_mm_slli_epi64(Z45QW,8);
    Y45QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFE000)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y45QW);Z23QW=_mm_slli_epi64(Z23QW,8);
    Y67QW = _mm_add_pd(_mm_mul_pd(X11QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFE00)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);Z67QW=_mm_slli_epi64(Z67QW,8);

    Y45QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFF80)])),_mm_setzero_pd(),(__m128d)Z01QW)),Y45QW);
    Y67QW = _mm_add_pd(_mm_mul_pd(X22QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFF8)])),_mm_setzero_pd(),(__m128d)Z45QW)),Y67QW);
    Y45QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFE0)])),_mm_setzero_pd(),(__m128d)Z23QW)),Y45QW);
    Y67QW = _mm_add_pd(_mm_mul_pd(X33QW,_mm_blendv_pd((__m128d)_mm_loadu_ps((float*)&(_V[__builtin_popcountll(M[ind]&0xFFFFFFFFFFFFFFFE)])),_mm_setzero_pd(),(__m128d)Z67QW)),Y67QW);

    //--------------------------------------------------------------------------
    _V += __builtin_popcountll(M[ind]);
    //--------------------------------------------------------------------------

    _mm_store_pd(&Y[Ri],Y01QW);
    _mm_store_pd(&Y[Ri+2],Y23QW);
    _mm_store_pd(&Y[Ri+4],Y45QW);
    _mm_store_pd(&Y[Ri+6],Y67QW);
  }
}

void popcountall(const unsigned char * __restrict M, unsigned * __restrict counts, size_t n)
{
	// only the last for bits counts in every location M[i]
	size_t nn = n/8;
	for(size_t i=0; i<nn; ++i)
	{
		counts[i*8] = __builtin_popcount(M[i*8] & 0x0F);
		counts[1+i*8] = __builtin_popcount(M[1+i*8] & 0x0F);
		counts[2+i*8] = __builtin_popcount(M[2+i*8] & 0x0F);
		counts[3+i*8] = __builtin_popcount(M[3+i*8] & 0x0F);
		counts[4+i*8] = __builtin_popcount(M[4+i*8] & 0x0F);
		counts[5+i*8] = __builtin_popcount(M[5+i*8] & 0x0F);
		counts[6+i*8] = __builtin_popcount(M[6+i*8] & 0x0F);
		counts[7+i*8] = __builtin_popcount(M[7+i*8] & 0x0F);
	}
	for(size_t i=nn*8; i<n; ++i)
	{
		counts[i] = __builtin_popcount(M[i] & 0x0F);
	}
}

void popcountall(const unsigned short * __restrict M, unsigned * __restrict counts, size_t n)
{
	size_t nn = n/8;
	for(size_t i=0; i<nn; ++i)
	{
		counts[i*8] = __builtin_popcount(M[i*8]);
		counts[1+i*8] = __builtin_popcount(M[1+i*8]);
		counts[2+i*8] = __builtin_popcount(M[2+i*8]);
		counts[3+i*8] = __builtin_popcount(M[3+i*8]);
		counts[4+i*8] = __builtin_popcount(M[4+i*8]);
		counts[5+i*8] = __builtin_popcount(M[5+i*8]);
		counts[6+i*8] = __builtin_popcount(M[6+i*8]);
		counts[7+i*8] = __builtin_popcount(M[7+i*8]);
	}
	for(size_t i=nn*8; i<n; ++i)
	{
		counts[i] = __builtin_popcount(M[i]);
	}
}

void popcountall(const uint64_t * __restrict M, unsigned * __restrict counts, size_t n)
{
	size_t nn = n/8;
	for(size_t i=0; i<nn; ++i)
	{
		counts[i*8] = __builtin_popcountl(M[i*8]);
		counts[1+i*8] = __builtin_popcountl(M[1+i*8]);
		counts[2+i*8] = __builtin_popcountl(M[2+i*8]);
		counts[3+i*8] = __builtin_popcountl(M[3+i*8]);
		counts[4+i*8] = __builtin_popcountl(M[4+i*8]);
		counts[5+i*8] = __builtin_popcountl(M[5+i*8]);
		counts[6+i*8] = __builtin_popcountl(M[6+i*8]);
		counts[7+i*8] = __builtin_popcountl(M[7+i*8]);
	}
	for(size_t i=nn*8; i<n; ++i)
	{
		counts[i] = __builtin_popcountl(M[i]);
	}
}



