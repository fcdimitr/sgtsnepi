#ifndef _BMSYM_H
#define _BMSYM_H

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

void symcsr(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lcmask, unsigned nlbits);

void symcsr(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lcmask, unsigned nlbits);

void symcsr(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lcmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, double * Y, unsigned lowmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, double * Y, unsigned lowmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, const double * __restrict XT, double * Y, double * YT, unsigned lowmask, unsigned nlbits);

void SSEsym(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, 
	const double * __restrict X, double * Y, unsigned lowmask, unsigned nlbits);

/* Symmetric CSB implementation
** Only upper triangle is stored
** top[i][0] gives the ith diagonal block for every i
** Since this class works only for symmetric (hence square) matrices,
** each compressed sparse block is (lowbits+1)x(lowbits+1) and ncsb = nbr = nbc
*/
template <class NT, class IT, unsigned TTDIM>
class BmSym
{
public:
	BmSym ():nz(0), n(0), ncsb(0) {}	// default constructor (dummy)

	BmSym (const BmSym<NT, IT,TTDIM> & rhs);			// copy constructor
	~BmSym();
	BmSym<NT,IT,TTDIM> & operator=(const BmSym<NT,IT,TTDIM> & rhs);	// assignment operator
	BmSym (Csc<NT, IT> & csc, int workers);
	
	ofstream & PrintStats(ofstream & outfile) const;
	ofstream & Dump(ofstream & outfile) const;
	IT colsize() const { return n;} 
	IT rowsize() const { return n;} 
	IT numregb() const { return nrb;}
	bool isPar() const { return ispar; }

private:
	typedef typename int_least_helper<TTDIM>::least MTYPE;

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
	IT nrbsum(IT d) const;

	IT ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size = nbr*(nbc+1)
	IT * bot;	// contains lower-order bits of the coordinate index, size nrb
	IT * scansum;	// prefix-sums on popcounts of masks, size nrb
	MTYPE * masks;	// array of masks, size nrb
	NT * num;	// contains numerical values, size nnz

	vector< pair<IT,NT> > diagonal;

	bool ispar;
	IT nz;		// # nonzeros 
	IT nrb;
	IT n;		// #{rows} = #{columns}
	IT blcrange;	// range indexed by one block

	IT ncsb; 	// #{block rows) = #{block cols}
	
	IT nlowbits;	// # lower order bits (for both rows and columns)
	IT nhighbits;
	IT highmask; 	// mask with the first log(n)/2 bits = 1 and the other bits = 0 
	IT lowmask;

	MortCompSym<IT> mortoncmp;	// comparison operator w.r.t. the (inverted N)-morton layout

	template <typename NU, typename IU, unsigned UUDIM>
	friend void bmsym_gespmv (const BmSym<NU, IU, UUDIM> & A, const NU * x, NU * y);
};


#include "friends.h"
#include "bmsym.cpp"
#endif

