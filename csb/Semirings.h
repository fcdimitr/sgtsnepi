
#ifndef _SEMIRINGS_H_
#define _SEMIRINGS_H_

#include <utility>
#include <climits>
#include <cmath>

#ifdef __APPLE__
#include <array>
#else
#include <tr1/array>
#endif

#include "promote.h"

template <typename T>
struct inf_plus{
  T operator()(const T& a, const T& b) const {
	T inf = std::numeric_limits<T>::max();
    	if (a == inf || b == inf){
      		return inf;
    	}
    	return a + b;
  }
};

// (+,*) on scalars
template <class T1, class T2>
struct PTSR
{
	typedef typename promote_trait<T1,T2>::T_promote T_promote;

	static T_promote add(const T1 & arg1, const T2 & arg2)
	{
		return (static_cast<T_promote>(arg1) +  
			static_cast<T_promote>(arg2) );
	}
	static T_promote multiply(const T1 & arg1, const T2 & arg2)
	{
		return (static_cast<T_promote>(arg1) * 
			static_cast<T_promote>(arg2) );
	}
	// y += ax overload with a=1
	static void axpy(const T2 & x, T_promote & y)
	{
		y += x;
	}
	
	static void axpy(T1 a, const T2 & x, T_promote & y)
	{
		y += a*x;
	}
};


template<int Begin, int End, int Step>
struct UnrollerL {
    template<typename Lambda>
    static void step(Lambda& func) {
        func(Begin);
        UnrollerL<Begin+Step, End, Step>::step(func);
    }
};

template<int End, int Step>
struct UnrollerL<End, End, Step> {
    template<typename Lambda>
    static void step(Lambda& func) {
		// base case is when Begin=End; do nothing
    }
};


// (+,*) on std:array's
template<class T1, class T2, unsigned D>
struct PTSRArray
{
	typedef typename promote_trait<T1,T2>::T_promote T_promote;

	// y <- a*x + y overload with a=1
	static void axpy(const array<T2, D> & b, array<T_promote, D> & c)
	{
		const T2 * __restrict barr =  b.data();
		T_promote * __restrict carr = c.data();
		__assume_aligned(barr, ALIGN);
		__assume_aligned(carr, ALIGN);

		#pragma simd
		for(int i=0; i<D; ++i)
		{
			carr[i] +=  barr[i];
		}
		// auto multadd = [&] (int i) { c[i] +=  b[i]; };
		// UnrollerL<0, D, 1>::step ( multadd );
	}
	
	// Todo: Do partial unrolling; this code will bloat for D > 32 
	static void axpy(T1 a, const array<T2,D> & b, array<T_promote,D> & c)
	{
		const T2 * __restrict barr =  b.data();
		T_promote * __restrict carr = c.data();
		__assume_aligned(barr, ALIGN);
		__assume_aligned(carr, ALIGN);

		#pragma simd
		for(int i=0; i<D; ++i)
		{
			carr[i] +=  a* barr[i];
		}	
		//auto multadd = [&] (int i) { carr[i] +=  a* barr[i]; };
		//UnrollerL<0, D, 1>::step ( multadd );	
	}
};

// (min,+) on scalars
template <class T1, class T2>
struct MPSR
{
	typedef typename promote_trait<T1,T2>::T_promote T_promote;

	static T_promote add(const T1 & arg1, const T2 & arg2)
	{
		return std::min<T_promote> 
		(static_cast<T_promote>(arg1), static_cast<T_promote>(arg2));
	}
	static T_promote multiply(const T1 & arg1, const T2 & arg2)
	{
		return inf_plus< T_promote > 
		(static_cast<T_promote>(arg1), static_cast<T_promote>(arg2));
	}
};


#endif
