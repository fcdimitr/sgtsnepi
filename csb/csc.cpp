#include "csc.h"
#include "utility.h"
#include <cassert>


template <class T, class ITYPE>
Csc<T,ITYPE>::Csc (ITYPE size, ITYPE rows, ITYPE cols, bool isSym): nz(size),m(rows),n(cols),issym(isSym),logicalnz(size)
{
	// Constructing empty Csc objects (size = 0) are not allowed.
	assert(size != 0 && n != 0);
	num = new T[nz];
	ir = new ITYPE[nz];
	jc = new ITYPE[n+1];
}


// copy constructor
template <class T, class ITYPE>
Csc<T,ITYPE>::Csc (const Csc<T, ITYPE> & rhs): nz(rhs.nz), m(rhs.m), n(rhs.n), issym(rhs.issym), logicalnz(rhs.logicalnz)
{
	if(nz > 0)
	{
		num = new T[nz];
		ir = new ITYPE[nz];

		for(ITYPE i=0; i< nz; ++i)		
			num[i]= rhs.num[i];
		for(ITYPE i=0; i< nz; ++i)		
			ir[i]= rhs.ir[i];
	}
	if ( n > 0)
	{
		jc	= new ITYPE[n+1];
		for(ITYPE i=0; i< n+1; i++)
			jc[i] = rhs.jc[i];
	}
}

template <class T, class ITYPE>
Csc<T,ITYPE> & Csc<T,ITYPE>::operator= (const Csc<T,ITYPE> & rhs)
{
	if(this != &rhs)		
	{
		if(nz > 0)	// if the existing object is not empty
		{
			// make it empty
			delete [] ir;
			delete [] num;
		}
		if(n > 0)
		{
			delete [] jc;
		}

		nz	= rhs.nz;
		m	= rhs.n;
		n	= rhs.n;
		issym 	= rhs.issym;
		logicalnz = rhs.logicalnz;
		if(rhs.nz > 0)	// if the copied object is not empty
		{
			num = new T[nz];
			ir = new ITYPE[nz];

			for(ITYPE i=0; i< nz; ++i)		
				num[i]= rhs.num[i];
			for(ITYPE i=0; i< nz; ++i)		
				ir[i]= rhs.ir[i];
		}
		if(rhs.n > 0)
		{
			jc = new ITYPE[n+1];
			for(ITYPE i=0; i< n+1; ++i)
				jc[i] = rhs.jc[i];
		}
	}
	return *this;
}


template <class T, class ITYPE>
Csc<T,ITYPE>::~Csc()
{
	if( nz > 0)
	{
		delete [] ir;
		delete [] num;
	}
	if ( n > 0)
	{
		delete [] jc;
	}
}


// Construct a Csc object from an array of "triple"s
// The symmetric case is resilient in the sense that it covers both cases
// (a) triples only contain the upper triangular part, or (b) the whole matrix
template <class T, class ITYPE>
Csc<T,ITYPE>::Csc(Triple<T, ITYPE> * triples, ITYPE size, ITYPE rows, ITYPE cols, bool isSym)
:nz(size),m(rows),n(cols),issym(isSym)
{
	// Constructing empty Csc objects (size = 0) are not allowed.
	assert(size != 0 && n != 0);

	num = new T[nz];
	ir = new ITYPE[nz];
	jc = new ITYPE[n+1];

	ITYPE * w = new ITYPE[n];	// workspace of size n (# of columns)

	for(ITYPE k = 0; k < n; ++k)
		w[k] = 0;

	if(issym)
	{	
		logicalnz = 0;
		for (ITYPE k = 0 ; k < nz ; ++k) 
		{
			if(triples[k].col >= triples[k].row)	// only the upper triangular part
			{
				if(triples[k].col > triples[k].row) 	
					++logicalnz;	// count each nonzero twice except for the diagonal
				ITYPE tmp =  triples[k].col;
				w [ tmp ]++ ;
				++logicalnz;
			}
		}
	}
	else
	{
		logicalnz = nz;
		for (ITYPE k = 0 ; k < nz ; ++k) 
		{
			ITYPE tmp =  triples[k].col;
			w [ tmp ]++ ;		// column counts (i.e, w holds the "col difference array")
		}
	}

	if(nz > 0)
	{
		jc[n] = CumulativeSum (w, n) ;		// cumulative sum of w
		for(ITYPE k = 0; k < n; ++k)
			jc[k] = w[k];

		ITYPE last;
		if(issym)
		{ 
			for(ITYPE k = 0; k < nz; ++k)
			{
				if(triples[k].col >= triples[k].row)	// only the upper triangular part
				{
					ir[ last = w[ triples[k].col ]++ ]  = triples[k].row ;    
					num[last] = triples[k].val ;
				}
			}
			nz = last + 1;		// actual number of nonzeros that are physically stored
			Resize(nz);
		}
		else
		{
			for (ITYPE k = 0 ; k < nz ; ++k)
			{
				ir[ last = w[ triples[k].col ]++ ]  = triples[k].row ;    
				num[last] = triples[k].val ;
			}
			assert(((last+1) == nz));
		}
	}
	delete [] w;
}


// Construct a Csc object from parallel arrays
template <class T, class ITYPE>
Csc<T,ITYPE>::Csc(ITYPE * ri, ITYPE * ci, T * val, ITYPE size, ITYPE rows, ITYPE cols, bool isSym)
:nz(size),m(rows),n(cols),issym(isSym)
{
	// Constructing empty Csc objects (size = 0) are not allowed.
	assert(size != 0 && n != 0);

	num = new T[nz];
	ir = new ITYPE[nz];
	jc = new ITYPE[n+1];

	ITYPE * w = new ITYPE[n];	// workspace of size n (# of columns)

	for(ITYPE k = 0; k < n; ++k)
		w[k] = 0;

	if(issym)
	{
		logicalnz = 0;
		for (ITYPE k = 0 ; k < nz ; ++k) 
		{
			if(ci[k] >= ri[k])	// only the upper part
			{
				if(ci[k] > ri[k]) 	
					++logicalnz;	// count each nonzero twice except for the diagonal
				ITYPE tmp = ci[k];
				w[ tmp ]++;
				++logicalnz;
			}
		}
	}
	else
	{	
		logicalnz = nz;
		for (ITYPE k = 0; k < nz; ++k)
		{
			ITYPE tmp = ci[k];
			w[ tmp ]++;		// column counts (i.e, w holds the "col difference array")
		}
	}
	if(nz > 0)
	{
		jc[n] = CumulativeSum (w, n) ;		// cumulative sum of w
		for(ITYPE k = 0; k < n; ++k)
			jc[k] = w[k];

		ITYPE last; 
		if(issym)
		{
			for (ITYPE k = 0 ; k < nz ; ++k) 
			{
				if(ci[k] >= ri[k])	// only the upper part
				{
					ir[ last = w[ ci[k] ]++ ]  = ri[k] ;    
					num[last] = val[k] ;
				}
			}	
			nz = last+1;	// actual nnz that are physically stored
			Resize(nz);
		}
		else
		{
		  cout << "nz=" << nz << " last=" << last << endl;
			for (ITYPE k = 0 ; k < nz ; ++k)
			{
				ir[ last = w[ ci[k] ]++ ]  = ri[k] ;    
				num[last] = val[k] ;
			}
			cout << "nz=" << nz << " last=" << last << endl;
			assert(((last+1) == nz));
		}
	}
	delete [] w;
}


// Resizes the maximum # nonzeros, doesn't change the matrix dimensions
template <class T, class ITYPE>
void Csc<T,ITYPE>::Resize(ITYPE nsize)
{
	if(nsize == nz)
	{
		// No need to do anything!
		return;
	}
	else if(nsize == 0)
	{
		delete [] num;
		delete [] ir;
		nz = 0;
		return;
	}

	T * tmpnum = num;
	ITYPE * tmpir = ir; 
	num = new T[nsize];
	ir = new ITYPE[nsize];

	if(nsize > nz)	// Grow it
	{
		for(ITYPE i=0; i< nz; ++i)	// copy all of the old elements
			num[i] = tmpnum[i];
		for(ITYPE i=0; i< nz; ++i)	// copy all of the old elements
			ir[i] = tmpir[i];
	}
	else			// Shrink it 
	{
		for(ITYPE i=0; i< nsize; ++i)	// copy only a portion of the old elements
			num[i] = tmpnum[i];
		for(ITYPE i=0; i< nsize; ++i)	// copy only a portion of the old elements
			ir[i] = tmpir[i];
	}
	delete [] tmpir;		// delete the memory pointed by previous pointers
	delete [] tmpnum;
	nz = nsize;
}
