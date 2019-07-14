#ifndef _BMCSB_H
#define _BMCSB_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>		// for std:accumulate()
#include <limits>		// C++ style numeric_limits<T>
#include "csc.h"
#include "mortoncompare.h"

using namespace std;

void SSEspmv(const double * __restrict V, const uint64_t * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits);

void SSEspmv(const double * __restrict V, const unsigned short * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits);

void SSEspmv(const double * __restrict V, const unsigned char * __restrict M, const unsigned * __restrict bot, const unsigned nrb, const double * __restrict X, double * Y, unsigned lcmask, unsigned lrmask, unsigned clbits);

template <class NT, class IT, unsigned TTDIM>
class BmCsb
{
public:
	BmCsb ():nz(0), m(0), n(0), nbc(0), nbr(0) {}	// default constructor (dummy)

	BmCsb (const BmCsb<NT, IT, TTDIM> & rhs);			// copy constructor
	~BmCsb();
	BmCsb<NT,IT,TTDIM> & operator=(const BmCsb<NT,IT,TTDIM> & rhs);	// assignment operator
	BmCsb (Csc<NT, IT> & csc, int workers);
	
	ofstream & PrintStats(ofstream & outfile) const;
	IT colsize() const { return n;} 
	IT rowsize() const { return m;} 
	IT numregb() const { return nrb;}
        IT numnonzeros() const { return nz; }
	bool isPar() const { return ispar; }

private:
	typedef typename int_least_helper<TTDIM>::least MTYPE;

	void Init(int workers, IT forcelogbeta = 0);

	void SubSpMV(IT * btop, IT bstart, IT bend, const NT * __restrict x, NT * __restrict suby, IT * __restrict sumscan) const;

	void BMult(IT** chunks, IT start, IT end, const NT * __restrict x, NT * __restrict y, IT ysize, IT * __restrict sumscan) const;


	void BlockPar(IT start, IT end, const NT * __restrict subx, NT * __restrict suby, 
					IT rangebeg, IT rangeend, IT cutoff, IT * __restrict sumscan) const;

	void SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val);

	IT ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size = nbr*(nbc+1)
	IT * bot;	// contains lower-order bits of the coordinate index, size nrb
	MTYPE * masks;	// array of masks, size nrb
	NT * num;	// contains numerical values, size nnz

	bool ispar;
	IT nz;		// # nonzeros 
	IT m;		// # rows
	IT n;		// # columns
	IT blcrange;	// range indexed by one block

	IT nbc;		// #{column blocks} = #{blocks in any block row}
	IT nbr; 	// #{block rows)
	IT nrb;		// #{register blocks}
	
	IT rowlowbits;	// # lower order bits for rows
	IT rowhighbits;
	IT highrowmask; // mask with the first log(m)/2 bits = 1 and the other bits = 0  
	IT lowrowmask;

	IT collowbits;	// # lower order bits for columns
	IT colhighbits;
	IT highcolmask; // mask with the first log(n)/2 bits = 1 and the other bits = 0  
	IT lowcolmask;

	MortonCompare<IT> mortoncmp;	// comparison operator w.r.t. the (inverted N)-morton layout

	template <typename NU, typename IU, unsigned UUDIM>
	friend void bmcsb_gespmv (const BmCsb<NU, IU, UUDIM> & A, const NU * x, NU * y);

	template <class CSB>
	friend float RowImbalance(const CSB & A);	// befriend any CSB instantiation	
};


#include "friends.h"
#include "bmcsb.cpp"	
#endif
