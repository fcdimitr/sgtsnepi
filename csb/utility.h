#ifndef _UTILITY_H
#define _UTILITY_H
#include <fstream>
#define __int64 long long
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <iostream>
#include <cmath>
#include <vector>
#include <mmintrin.h>  //  MMX
#include <xmmintrin.h> //  SSE
#include <emmintrin.h> //  SSE 2  
#include <pmmintrin.h> //  SSE 3

#include <numeric>

using namespace std;

#include <cilk/cilk_api.h>
#include <cilk/cilk.h>
#define SYNCHED __cilkrts_synched()
#define DETECT __cilkscreen_enable_checking()
#define ENDDETECT __cilkscreen_disable_checking()
#define WORKERS __cilkrts_get_nworkers()

#ifdef BWTEST
	#define UNROLL 100
#else
	#define UNROLL 1
#endif

#ifndef CILK_STUB
#ifdef __cplusplus
extern "C" {
#endif
/*
 * __cilkrts_synched
 *
 * Allows an application to determine if there are any outstanding
 * children at this instant. This function will examine the current
 * full frame to determine this.
 */

CILK_EXPORT __CILKRTS_NOTHROW
int __cilkrts_synched(void);

#ifdef __cplusplus
} // extern "C"
#endif
#else /* CILK_STUB */
/* Stubs for the api functions */
#define __cilkrts_synched() (1)
#endif /* CILK_STUB */

#ifdef STATS
	#include <cilk/reducer_opadd.h>
	cilk::reducer_opadd<__int64> blockparcalls;
	cilk::reducer_opadd<__int64> subspmvcalls;
	cilk::reducer_opadd<__int64> atomicflops;
#endif

void * address;
void * base;

using namespace std;

//  convert category to type 
  template< int Category > struct int_least_helper {}; // default is empty
  template<> struct int_least_helper<8> { typedef uint64_t least; };		// 8x8 blocks require 64-bit bitmasks
  template<> struct int_least_helper<4> { typedef unsigned short least; };	// 4x4 blocks require 16-bit bitmasks
  template<> struct int_least_helper<2> { typedef unsigned char least; };	// 2x2 blocks require 4-bit bitmasks, so we waste half of the array here

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


const unsigned char masktable4[4] = { 0x08, 0x04, 0x02, 0x01 };	// mask for 2x2 register blocks


template <typename MTYPE>
MTYPE GetMaskTable(unsigned int index)
{
	return 0;
} 


template <>
uint64_t GetMaskTable<uint64_t>(unsigned int index)
{
	return masktable64[index];
} 

template <>
unsigned short GetMaskTable<unsigned short>(unsigned int index)
{
	return masktable16[index];
} 


template <>
unsigned char GetMaskTable<unsigned char>(unsigned int index)
{
	return masktable4[index];
} 

#ifndef RHSDIM
#define RHSDIM 1
#endif
#define BALANCETH 2.0
//#define BALANCETH 1.0
#define RBDIM 8
#define RBSIZE (RBDIM*RBDIM)		// size of a register block (8x8 in this case)
#define SLACKNESS 8
#define KBYTE 1024
#define L2SIZE (256*KBYTE / RHSDIM)	// less than half of the L2 Cache (L2 should hold x & y at the same time) - scaled back by RHSDIM
#define CLSIZE 64			// cache line size

/* Tuning Parameters */
#define BREAKEVEN 4		// A block (or subblock) with less than (BREAKEVEN * dimension) nonzeros won't be parallelized
#define MINNNZTOPAR 128		// A block (or subblock) with less than MINNNZTOPAR nonzeros won't be parallelized
#define BREAKNRB (8/RBDIM)	// register blocked version of BREAKEVEN
#define MINNRBTOPAR (256/RBDIM)	// register blocked version of MINNNZPAR
#define LOGSERIAL 15
#define ROLLING 20

#define EPSILON 0.0001
#define REPEAT 10

// "absolute" difference macro that has no possibility of unsigned wrap
#define absdiff(x,y)   ( (x) > (y) ? (x-y) : (y-x))


unsigned rmasks[32] = { 0x00000001, 0x00000002, 0x00000004, 0x00000008,
			0x00000010, 0x00000020, 0x00000040, 0x00000080,  
			0x00000100, 0x00000200, 0x00000400, 0x00000800,  
			0x00001000, 0x00002000, 0x00004000, 0x00008000,  
			0x00010000, 0x00020000, 0x00040000, 0x00080000,  
			0x00100000, 0x00200000, 0x00400000, 0x00800000,  
			0x01000000, 0x02000000, 0x04000000, 0x08000000,  
			0x10000000, 0x20000000, 0x40000000, 0x80000000 };  


void popcountall(const uint64_t * __restrict M, unsigned * __restrict count, size_t size);
void popcountall(const unsigned short * __restrict M, unsigned * __restrict count, size_t size);
void popcountall(const unsigned char * __restrict M, unsigned * __restrict count, size_t size);


template <typename T>
void printhistogram(const T * scansum, size_t size, unsigned bins)
{
	ofstream outfile;
	outfile.open("hist.csv");
	vector<T> hist(bins);	// an STD-vector is zero initialized
	for(size_t i=0; i< size; ++i)
		hist[scansum[i]]++;

	outfile << "Fill_ratio" << "," << "count" << endl;
	for(size_t i=0; i< bins; ++i)
	{
		outfile << static_cast<float>(i) / bins  << "," << hist[i] << "\n";
	}
}

struct thread_data
{
	unsigned sum;
	unsigned * beg;
	unsigned * end;
};
	
unsigned int highestbitset(unsigned __int64 v);

template <typename MTYPE>
unsigned prescan(unsigned * a, MTYPE * const M, int n)	
{
	unsigned * end = a+n;
	unsigned * _a = a;	
	MTYPE * __restrict _M = M;
	unsigned int lgn;
	unsigned sum = 0;
	while ((lgn = highestbitset(n)) > LOGSERIAL)
	{
		unsigned _n = rmasks[lgn];	// _n: biggest power of two that is less than n
		int numthreads = SLACKNESS*WORKERS;
		thread_data * thdatas = new thread_data[numthreads];
		unsigned share = _n/numthreads;
		cilk_for(int t=0; t < numthreads; ++t)
		{
			popcountall(_M+t*share, _a+t*share, ((t+1)==numthreads)?(_n-t*share):share);
			thdatas[t].sum = 0;
			thdatas[t].beg = _a + t*share;
			thdatas[t].end = _a + (((t+1)==numthreads)?_n:((t+1)*share));
			thdatas[t].sum = accumulate(thdatas[t].beg, thdatas[t].end, thdatas[t].sum);
		}
		for(int t=0; t<numthreads; ++t)
		{
			unsigned temp = thdatas[t].sum;
			thdatas[t].sum = sum;
			sum += temp;
		}
		cilk_for(int tt=0; tt<numthreads; ++tt)
		{				
			unsigned * beg = thdatas[tt].beg;
			unsigned * end = thdatas[tt].end;	
			unsigned locsum = thdatas[tt].sum;

			while(beg != end)
			{
				unsigned temp = *beg;
				*beg++ = locsum;   // changing the value of (*beg) changes the corresponding aliased pointer _a as well
				locsum += temp; 
			}
		}
		_a += _n;	// move the pointer on a
		_M += _n;	// move the pointer on M
		n  &=  ~_n;	// clear the highest bit
		delete [] thdatas;
	}
	popcountall(_M, _a, end-(_a));
	while(_a != end)
	{
		unsigned temp = *_a;
		*_a = sum;
		sum += temp; 
		_a++;
	}
	return sum;
}

extern "C"
unsigned char *aligned_malloc( uint64_t size ) {
  unsigned char *ret_ptr = (unsigned char *)malloc( size + 16 );
  int temp = (unsigned long)ret_ptr & 0xF;
  int shift = 16 - temp;
  ret_ptr += shift;
  ret_ptr[ -1 ] = shift;
  return( ret_ptr );
}

extern "C"
void aligned_free( unsigned char *ptr ) {
  ptr -= ptr[ -1 ];
  free( ptr );
}


template <typename ITYPE>
ITYPE CumulativeSum (ITYPE * arr, ITYPE size)
{
    ITYPE prev;
    ITYPE tempnz = 0 ;
    for (ITYPE i = 0 ; i < size ; ++i)
    {
		prev = arr[i];
		arr[i] = tempnz;
		tempnz += prev ;	
    }
    return (tempnz) ;		    // return sum
}


template <typename T>
T machineEpsilon()
{	
	T machEps = 1.0;
 	do {
       		machEps /= static_cast<T>(2.0);
       		// If next epsilon yields 1, then break, because current
       		// epsilon is the machine epsilon.
    	}
    	while ((T)(static_cast<T>(1.0) + (machEps/static_cast<T>(2.0))) != 1.0);
 
    	return machEps;
}


template<typename _ForwardIter, typename T>
void iota(_ForwardIter __first, _ForwardIter __last, T __value)
{
	while (__first != __last)
     		*__first++ = __value++;
}
	
template<typename T, typename I>
T ** allocate2D(I m, I n)
{
	T ** array = new T*[m];
	for(I i = 0; i<m; ++i) 
		array[i] = new T[n]();
	return array;
}

template<typename T, typename I>
void deallocate2D(T ** array, I m)
{
	for(I i = 0; i<m; ++i) 
		delete [] array[i];
	delete [] array;
}


template < typename T >
struct absdiff : binary_function<T, T, T>
{
        T operator () ( T const &arg1, T const &arg2 ) const
        {
                using std::abs;
                return abs( arg1 - arg2 );
        }
};



template <int D>
void MultAdd(double & a, const double & b, const double & c)
{
	for(int i=0; i<D; i++)
	{
		a += b * c;
	}	
	
}

// bit interleave x and y, and return result
// only the lower order bits of x and y are assumed valid
template <typename ITYPE>
ITYPE BitInterleaveLow(ITYPE x, ITYPE y)
{
	ITYPE z = 0; // z gets the resulting Morton Number.
	int ite = sizeof(z) * CHAR_BIT / 2;

	for (int i = 0; i < ite; ++i) 
	{
		// bitwise shift operations have precedence over bitwise OR and AND
  		z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1);
	}
	return z;
}

// bit interleave x and y, and return result z (which is twice in size)
template <typename ITYPE, typename OTYPE>
OTYPE BitInterleave(ITYPE x, ITYPE y)
{
	OTYPE z = 0; // z gets the resulting Morton Number.
	int ite = sizeof(x) * CHAR_BIT;

	for (int i = 0; i < ite; ++i) 
	{
		// bitwise shift operations have precedence over bitwise OR and AND
  		z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1);
	}
	return z;
}

template <unsigned BASE>
inline unsigned IntPower(unsigned exponent)
{
	unsigned i = 1; 
	unsigned power = 1;

	while ( i <= exponent ) 
	{
		power *= BASE;
		i++;
	}
	return power;
}

template <>
inline unsigned IntPower<2>(unsigned exponent)
{	
	return rmasks[exponent];
}



// T should be uint32, uint64, int32 or int64; force concept requirement
template <typename T>
bool IsPower2(T x)
{
	return ( (x>0) && ((x & (x-1)) == 0));
}

unsigned int nextpoweroftwo(unsigned int v)
{
	// compute the next highest power of 2 of 32(or 64)-bit n
	// essentially does 1 << (lg(n - 1)+1).

	unsigned int n = v-1;

	// any "0" that is immediately right to a "1" becomes "1" (post: any zero has at least two "1"s to its left) 
	n |= n >> 1;

	// turn two more adjacent "0" to "1" (post: any zero has at least four "1"s to its left)
	n |= n >> 2;
	n |= n >> 4;	// post: any zero has at least 8 "1"s to its left
	n |= n >> 8;	// post: any zero has at least 16 "1"s to its left
	n |= n >> 16;	// post: any zero has at least 32 "1"s to its left

	return ++n;
}

// 64-bit version
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(unsigned __int64 v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const unsigned __int64 b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

__int64 highestbitset(__int64 v)
{
	if(v < 0)
	{
		cerr << "Indices can not be negative, aborting..." << endl;
		return -1;
	}
	else
	{
		unsigned __int64 uv = static_cast< unsigned __int64 >(v);
		unsigned __int64 ur = highestbitset(uv);
		return static_cast< __int64 > (ur);
	}
}
		
// 32-bit version 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(unsigned int v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const unsigned int S[] = {1, 2, 4, 8, 16};
	int i;

	unsigned int r = 0; 
	for (i = 4; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

int highestbitset(int v)
{
	if(v < 0)
	{
		cerr << "Indices can not be negative, aborting..." << endl;
		return -1;
	}
	else
	{	
		unsigned int uv = static_cast< unsigned int> (v);
		unsigned int ur = highestbitset(uv);
		return static_cast< int > (ur); 
	}
}

/* This function will return n % d.
   d must be one of: 1, 2, 4, 8, 16, 32, … */
inline unsigned int getModulo(unsigned int n, unsigned int d)
{
	return ( n & (d-1) );
} 

// Same requirement (d=2^k) here as well
inline unsigned int getDivident(unsigned int n, unsigned int d)
{
	while((d = d >> 1))
		n = n >> 1;
	return n;
}

#endif

