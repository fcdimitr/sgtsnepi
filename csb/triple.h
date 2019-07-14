#ifndef _TRIPLE_H_
#define _TRIPLE_H_

#include <iostream>
#include <functional>
#include "utility.h"
#include "spvec.h"
using namespace std;

template <class T, class ITYPE>
struct Triple
{
	ITYPE row;	//  row index
	ITYPE col;	//  col index
	T val;		//  value
};	

template <class T, class ITYPE>
struct ColSortCompare:	// struct instead of class so that operator() is public
	public binary_function< Triple<T, ITYPE>, Triple<T, ITYPE>, bool >	// (par1, par2, return_type)
	{
		inline bool operator()(const Triple<T, ITYPE> & lhs, const Triple<T, ITYPE> & rhs) const
		{
			if(lhs.col  == rhs.col)
			{
				return lhs.row < rhs.row;
			}
			else
			{
				return lhs.col < rhs.col;
			}
		}
	};

template <class T, class ITYPE>
struct RowSortCompare:	// struct instead of class so that operator() is public
	public binary_function< Triple<T, ITYPE>, Triple<T, ITYPE>, bool >	// (par1, par2, return_type)
	{
		inline bool operator()(const Triple<T, ITYPE> & lhs, const Triple<T, ITYPE> & rhs) const
		{
			if(lhs.row  == rhs.row)
			{
				return lhs.col < rhs.col;
			}
			else
			{
				return lhs.row < rhs.row;
			}
		}
	};

template <class T, class ITYPE, class OTYPE>
struct BitSortCompare:	// struct instead of class so that operator() is public
	public binary_function< Triple<T, ITYPE>, Triple<T, ITYPE>, bool >	// (par1, par2, return_type)
	{
		inline bool operator()(const Triple<T, ITYPE> & lhs, const Triple<T, ITYPE> & rhs) const
		{
			return BitInterleave<ITYPE, OTYPE>(lhs.row, lhs.col) < BitInterleave<ITYPE, OTYPE>(rhs.row, rhs.col);
		}
	};

template <typename T, typename ITYPE> 
void triples_gaxpy(Triple<T, ITYPE> * triples, Spvec<T, ITYPE> & x, Spvec<T, ITYPE> & y, ITYPE nnz)
{
	for(ITYPE i=0; i< nnz; ++i)		
	{
		y [triples[i].row] += triples[i].val * x [triples[i].col] ;
	}
};


#endif

