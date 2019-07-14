#ifndef _CSBSYM_H
#define _CSBSYM_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>		// for std:accumulate()
#include <limits>		// C++ style numeric_limits<T>
#include <ostream>
#include <iterator>
#include "csc.h"
#include "mortoncompare.h"

using namespace std;


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

/* Symmetric CSB implementation
** Only upper triangle is stored
** top[i][0] gives the ith diagonal block for every i
** Since this class works only for symmetric (hence square) matrices,
** each compressed sparse block is (lowbits+1)x(lowbits+1) and ncsb = nbr = nbc
*/
template <class NT, class IT>
class CsbSym
{
public:
	CsbSym ():nz(0), n(0), ncsb(0) {}	// default constructor (dummy)

	CsbSym (const CsbSym<NT, IT> & rhs);			// copy constructor
	~CsbSym();
	CsbSym<NT,IT> & operator=(const CsbSym<NT,IT> & rhs);	// assignment operator
	CsbSym (Csc<NT, IT> & csc, int workers);
	
	ofstream & PrintStats(ofstream & outfile) const;
	ofstream & Dump(ofstream & outfile) const;
	IT colsize() const { return n;} 
	IT rowsize() const { return n;} 
	bool isPar() const { return ispar; }

private:
	void Init(int workers, IT forcelogbeta = 0);
	void SeqSpMV(const NT * __restrict x, NT * __restrict y) const;
	void BMult(IT** chunks, IT start, IT end, const NT * __restrict x, NT * __restrict y, IT ysize) const;

	void BlockPar(IT start, IT end, const NT * __restrict subx, const NT * __restrict subx_mirror, 
			NT * __restrict suby, NT * __restrict suby_mirror, IT rangebeg, IT rangeend, IT cutoff) const;
	void BlockTriPar(IT start, IT end, const NT * __restrict subx, NT * __restrict suby, IT rangebeg, IT rangeend, IT cutoff) const;

	void SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val);
	void DivideIterationSpace(IT * & lspace, IT * & rspace, IT & lsize, IT & rsize, IT size, IT d) const;

	void MultAddAtomics(NT * __restrict y, const NT * __restrict x, const IT d) const;
	void MultDiag(NT * __restrict y, const NT * __restrict x, const IT d) const;
	void MultMainDiag(NT * __restrict y, const NT * __restrict x) const;

	float Imbalance(IT d) const;
	IT nzsum(IT d) const;

	IT ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size = nbr*(nbc+1)
	IT * bot;	// contains lower-order bits of the coordinate index, size nnz
	NT * num;	// contains numerical values, size nnz

	vector< pair<IT,NT> > diagonal;

	bool ispar;
	IT nz;		// # nonzeros 
	IT n;		// #{rows} = #{columns}
	IT blcrange;	// range indexed by one block

	IT ncsb; 	// #{block rows) = #{block cols}
	
	IT nlowbits;	// # lower order bits (for both rows and columns)
	IT nhighbits;
	IT highmask; 	// mask with the first log(n)/2 bits = 1 and the other bits = 0 
	IT lowmask;

	MortCompSym<IT> mortoncmp;	// comparison operator w.r.t. the (inverted N)-morton layout

	template <typename NU, typename IU>
	friend void csbsym_gespmv (const CsbSym<NU, IU> & A, const NU * x, NU * y);
};


#include "friends.h"
#include "csbsym.cpp"
#endif

