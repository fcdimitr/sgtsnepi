#ifndef _BICSB_H
#define _BICSB_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>		// for std:accumulate()
#include <limits>		// C++ style numeric_limits<T>
#include <tuple>
#include "csc.h"
#include "mortoncompare.h"

using namespace std;

// CSB variant where nonzeros "within each block" are sorted w.r.t. the bit-interleaved order
// Implementer's (Aydin) notes:
//	- to ensure correctness in BlockPar, we use square blocks (lowcolmask = highcolmask)
template <class NT, class IT>
class BiCsb
{
public:
	BiCsb ():nz(0), m(0), n(0), nbc(0), nbr(0) {}	// default constructor (dummy)

	BiCsb (IT size,IT rows, IT cols, int workers);
	BiCsb (IT size,IT rows, IT cols, IT * ri, IT * ci, NT * val, int workers, IT forcelogbeta = 0);

	BiCsb (const BiCsb<NT, IT> & rhs);			// copy constructor
	~BiCsb();
	BiCsb<NT,IT> & operator=(const BiCsb<NT,IT> & rhs);	// assignment operator
	BiCsb (Csc<NT, IT> & csc, int workers, IT forcelogbeta = 0);
	
	ofstream & PrintStats(ofstream & outfile) const;
	IT colsize() const { return n;} 
	IT rowsize() const { return m;}
        IT getNbc()  const { return nbc;}
        IT getNbr()  const { return nbr;}
        IT getBeta()  const { return rowlowbits;}
    IT numnonzeros() const { return nz; }
	bool isPar() const { return ispar; }

        /* function to output top level statistics to CSV (in dense format) */
        ofstream & PrintTopLevel(ofstream & outfile) const;

        /* function to output top level statistics to CSV (in sparse format) */
        ofstream & PrintTopLevelSparse(ofstream & outfile) const;
        
private:
	void Init(int workers, IT forcelogbeta = 0);

	template <typename SR, typename RHS, typename LHS>	
	void SubSpMV(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const;

        template <typename SR, typename RHS, typename LHS>	
	void SubSpMV_tar(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const;

        template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEcost(IT * btop, IT bstart, IT bend,
                           const RHS * __restrict x,
                           LHS * __restrict suby,
                           IT rhi, int dim,
                           double alpha, double zeta) const;

        
	template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel(IT * btop, IT bstart, IT bend,
			     const RHS * __restrict x,
			     LHS * __restrict suby,
			     IT rhi) const;

        template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel1D(IT * btop, IT bstart, IT bend,
			     const RHS * __restrict x,
			     LHS * __restrict suby,
			     IT rhi) const;

        template <typename SR, typename RHS, typename LHS>	
          void SubtSNEkernel2D(IT * btop, IT bstart, IT bend,
			     const RHS * __restrict x,
			     LHS * __restrict suby,
			     IT rhi) const;


        template <typename SR, typename RHS, typename LHS>	
          void SubtSNEkernel4D(IT * btop, IT bstart, IT bend,
			     const RHS * __restrict x,
			     LHS * __restrict suby,
			     IT rhi) const;

        template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel_tar(IT * btop, IT bstart, IT bend,
			     const RHS * __restrict x,
			     LHS * __restrict suby,
			     IT rhi) const;
	
	template <typename SR, typename RHS, typename LHS>
	  void SubtSNEkernel(IT * __restrict btop, IT bstart, IT bend,
			     const RHS * __restrict x_row,
			     const RHS * __restrict x_col,
			     LHS * __restrict suby,
			     IT rhi) const;
        
	template <typename SR, typename RHS, typename LHS>	
	void SubSpMVTrans(IT col, IT rowstart, IT rowend, const RHS * __restrict x, LHS * __restrict suby) const;
    
    template <typename SR, typename RHS, typename LHS>
    void SubSpMVTrans(const vector< tuple<IT,IT,IT> > & chunk, const RHS * __restrict x, LHS * __restrict suby) const;

	template <typename SR, typename RHS, typename LHS>
	void BMult(IT** chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const;

	template <typename SR, typename RHS, typename LHS>	
	void BTransMult(vector< vector< tuple<IT,IT,IT> > * > & chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const;

	template <typename SR, typename RHS, typename LHS>	
	void BlockPar(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
					IT rangebeg, IT rangeend, IT cutoff) const;

	template <typename SR, typename RHS, typename LHS>	
	void BlockParT(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
					IT rangebeg, IT rangeend, IT cutoff) const;

	void SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val);

	IT ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size ~= ntop+1
	IT * bot;	// contains lower-order bits of the coordinate index, size nnz 
	NT * num;	// contains numerical values, size nnz

	bool ispar;
	IT nz;		// # nonzeros
	IT m;		// # rows
	IT n;		// # columns
	IT blcrange;	// range indexed by one block

	IT nbc;		// #{column blocks} = #{blocks in any block row}
	IT nbr; 	// #{block rows)
	
	IT rowlowbits;	// # lower order bits for rows
	IT rowhighbits;
	IT highrowmask; // mask with the first log(m)/2 bits = 1 and the other bits = 0  
	IT lowrowmask;

	IT collowbits;	// # lower order bits for columns
	IT colhighbits;
	IT highcolmask; // mask with the first log(n)/2 bits = 1 and the other bits = 0  
	IT lowcolmask;

	MortonCompare<IT> mortoncmp;	// comparison operator w.r.t. the N-morton layout

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_gespmv (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_gespmv_tar (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_gespmvt (const BiCsb<NU, IU> & A, const RHS * __restrict x, LHS * __restrict y);

	template <class CSB>
	friend float RowImbalance(const CSB & A);	// just befriend the BiCsb instantiation

	template <typename NU, typename IU>
	friend float ColImbalance(const BiCsb<NU, IU> & A);

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
          friend void bicsb_tsne (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
          friend void bicsb_tsne4D (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);
        
        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
          friend void bicsb_tsne2D (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
          friend void bicsb_tsne1D (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_tsne_tar (const BiCsb<NU, IU> & A, const RHS * x, LHS * y);

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	  friend void bicsb_tsne (const BiCsb<NU, IU> & A,
				  const RHS * x_row,
				  const RHS * x_col, LHS * y);

        template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
          friend void bicsb_tsne_cost (const BiCsb<NU, IU> & A,
                                       const RHS * x, LHS * y, int dim,
                                       double alpha, double zeta);
};


// Partial template specialization
template <class IT>
class BiCsb<bool,IT>
{
public:
	BiCsb ():nz(0), m(0), n(0), nbc(0), nbr(0) {}	// default constructor (dummy)

	BiCsb (IT size,IT rows, IT cols, int workers);
	BiCsb (IT size,IT rows, IT cols, IT * ri, IT * ci, int workers, IT forcelogbeta = 0);	

	BiCsb (const BiCsb<bool, IT> & rhs);			// copy constructor
	~BiCsb();
	BiCsb<bool,IT> & operator=(const BiCsb<bool,IT> & rhs);	// assignment operator
	
	template <typename NT>
	BiCsb (Csc<NT, IT> & csc, int workers);
	
	IT colsize() const { return n;} 
	IT rowsize() const { return m;}
    IT numnonzeros() const { return nz; }
	bool isPar() const { return ispar; }

private:
	void Init(int workers, IT forcelogbeta = 0);

	template <typename SR, typename RHS, typename LHS>
	void SubSpMV(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const;
	template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const;

        template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel2D(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const;

        template <typename SR, typename RHS, typename LHS>	
	  void SubtSNEkernel1D(IT * btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const;

	template <typename SR, typename RHS, typename LHS>	
	void SubSpMVTrans(IT col, IT rowstart, IT rowend, const RHS * __restrict x, LHS * __restrict suby) const;
    
    template <typename SR, typename RHS, typename LHS>
    void SubSpMVTrans(const vector< tuple<IT,IT,IT> > & chunk, const RHS * __restrict x, LHS * __restrict suby) const;

	template <typename SR, typename RHS, typename LHS>	
	void BMult(IT ** chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const;

	template <typename SR, typename RHS, typename LHS>		
	void BTransMult(vector< vector< tuple<IT,IT,IT> > * > & chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const;

	template <typename SR, typename RHS, typename LHS>		
	void BlockPar(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
					IT rangebeg, IT rangeend, IT cutoff) const;
	
	template <typename SR, typename RHS, typename LHS>		
	void BlockParT(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
					IT rangebeg, IT rangeend, IT cutoff) const;

	void SortBlocks(pair<IT, pair<IT,IT> > * pairarray);

	IT ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size ~= ntop+1
	IT * bot;	// contains lower-order bits of the coordinate index, size nnz 

	bool ispar;
	IT nz;		// # nonzeros
	IT m;		// # rows
	IT n;		// # columns
	IT blcrange;	// range indexed by one block

	IT nbc;		// #{column blocks} = #{blocks in any block row}
	IT nbr; 	// #{block rows)
	
	IT rowlowbits;	// # lower order bits for rows
	IT rowhighbits;
	IT highrowmask;	// mask with the first log(m)/2 bits = 1 and the other bits = 0  
	IT lowrowmask;

	IT collowbits;	// # lower order bits for columns
	IT colhighbits;
	IT highcolmask;  	// mask with the first log(n)/2 bits = 1 and the other bits = 0  
	IT lowcolmask;

	MortonCompare<IT> mortoncmp;	// comparison operator w.r.t. the N-morton layout

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_gespmv (const BiCsb<NU, IU> & A, const RHS * __restrict x, LHS * __restrict y);

	template <typename SR, typename NU, typename IU, typename RHS, typename LHS>
	friend void bicsb_gespmvt (const BiCsb<NU, IU> & A, const RHS * __restrict x, LHS * __restrict y);

	template <class CSB>
	friend float RowImbalance(const CSB & A);	// befriend any CSB instantiation	

	template <typename NU, typename IU>
	friend float ColImbalance(const BiCsb<NU, IU> & A);
};

#include "friends.h"
#include "bicsb.cpp"	
#endif


/*------------------------------------------------------------
 *
 * AUTHORS
 *
 *   Dimitris Floros                         fcdimitr@auth.gr
 *
 * VERSION
 *
 *   0.3 - December 16, 2017
 *
 * CHANGELOG
 *
 *   0.3 (Dec 16, 2017) - Dimitris
 *      * incorporated TAR and TAR+ codes
 *      
 *   0.2 (Dec 12, 2017) - Dimitris
 *      * added sparse output of top-level statistics
 *  
 *   0.1 (Dec 08, 2017) - Dimitris
 *      * added custom function to get top-level statistics
 *
 * ----------------------------------------------------------*/
