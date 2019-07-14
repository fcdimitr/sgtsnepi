#ifndef _SPVEC_H_
#define _SPVEC_H_

#include "csc.h"
#include "bicsb.h"
#include "matmul.h"
#include "Semirings.h"

template <class T, class ITYPE>
class Spvec
{
public:
	Spvec (): n(0) {};
	Spvec (ITYPE dim);				
	Spvec (T * darr, ITYPE dim);
	Spvec (const Spvec<T,ITYPE> & rhs);		
	~Spvec();
	Spvec<T,ITYPE> & operator=(const Spvec<T, ITYPE> & rhs);	

	T& operator[] (const ITYPE nIndex)
	{
		return arr[nIndex];
	}

	//! Delayed evaluations using compositors for SpMV operation...  y <- y + Ax
	Spvec<T,ITYPE> & operator+=(const Matmul< Csc<T, ITYPE>, Spvec<T,ITYPE> > & matmul);	
	Spvec<T,ITYPE> & operator+=(const Matmul< BiCsb<T, ITYPE>, Spvec<T,ITYPE> > & matmul);

	void fillzero();
	void fillrandom();
	void fillone()
	{
		std::fill(arr,arr+n, static_cast<T>(1.0));
	}
	void fillfota()
	{
		for(ITYPE i =0; i<n; ++i)
			arr[i] = (i+1) * static_cast<T>(1.0);
	}

	ITYPE size() const { return n-padding;} // return the real size
	T * getarr(){ return arr;} 

private:
	T * arr;
	ITYPE n;
	ITYPE padding;
};

#include "spvec.cpp"
#endif

