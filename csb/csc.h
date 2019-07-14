#ifndef _CSC_H_
#define _CSC_H_

#include "triple.h"
#include <iterator>
#include <array>

using namespace std;


template <class T, class ITYPE>
struct Triple;

template <class T, class ITYPE>
class Csc
{
public:
	Csc ():nz(0), m(0), n(0), logicalnz(0), issym(false) {}				// default constructor
	Csc (ITYPE size,ITYPE rows, ITYPE cols, bool isSym=false);
	Csc (const Csc<T, ITYPE> & rhs);		// copy constructor
	~Csc();
	Csc<T, ITYPE> & operator=(const Csc<T, ITYPE> & rhs);	// assignment operator
	Csc (Triple<T, ITYPE> * triples, ITYPE size, ITYPE rows, ITYPE cols, bool isSym=false);
	Csc (ITYPE * ri, ITYPE * ci, T * val, ITYPE size, ITYPE rows, ITYPE cols, bool isSym=false);

	// we have to use another function because the compiler will reject another constructor with the same signature
	void SetPointers (ITYPE * colpointers, ITYPE * rowindices, T * vals, ITYPE size, ITYPE rows, ITYPE cols, bool fortran)	
	{
		jc = colpointers;
		ir = rowindices;
		num = vals;
		nz = size;
		m = rows;
		n = cols;
		issym = false;
		logicalnz = size;

		if(fortran)
		{
			transform(jc, jc+n+1, jc, bind2nd(minus<ITYPE>(),1));
			transform(ir, ir+nz, ir, bind2nd(minus<ITYPE>(),1));
		}
	}

        // symmetric pointer initialization
        void SetPointersSym (ITYPE * colpointers, ITYPE * rowindices, T * vals, ITYPE size, ITYPE sizeNz, ITYPE rows, ITYPE cols, bool fortran)	
	{
          jc = colpointers;
          ir = rowindices;
          num = vals;
          nz = size;
          m = rows;
          n = cols;
          issym = true;
          logicalnz = sizeNz;

          if(fortran)
            {
              transform(jc, jc+n+1, jc, bind2nd(minus<ITYPE>(),1));
              transform(ir, ir+nz, ir, bind2nd(minus<ITYPE>(),1));
		}
	}

	ITYPE colsize() const { return n;} 
	ITYPE rowsize() const { return m;} 
	ITYPE * getjc() const { return jc;} 
	ITYPE * getir() const { return ir;} 
	T * getnum() const { return num;} 
	ITYPE getlogicalnnz() const
	{
		return logicalnz;
	}

        // function to print CSC stats for debugging
        void printStats() const {
          printf("  nz        = %d\n"      , nz           );
          printf("  m         = %d\n"      , m            );
          printf("  n         = %d\n"      , n            );
          printf("  issym     = %d\n"      , issym        );
          printf("  logicalnz = %d\n"      , logicalnz    );

          for (int j = 0; j < n; j++) 
            for (int i = jc[j]; i < jc[j+1]; i++)
              printf("    A[%d, %d] = %g\n", ir[i], j, num[i] );
          
        }

private:
	void Resize(ITYPE nsize);
	bool issym;

	ITYPE * jc ;	// col pointers, size n+1
	ITYPE * ir;		// row indices, size nnz 
	T * num;	// numerical values, size nnz 
	
	ITYPE logicalnz;
	ITYPE nz;
	ITYPE m;		//  number of rows
	ITYPE n;		//  number of columns

	template <class U, class UTYPE>
	friend class CsbSym;
	template <class U, class UTYPE>
	friend class BiCsb;
	template <class U, class UTYPE, unsigned UDIM>
	friend class BmCsb;
	template <class U, class UTYPE, unsigned UDIM>
	friend class BmSym;

	template <typename U, typename UTYPE>
	friend void csc_gaxpy (const Csc<U, UTYPE> & A, U * x, U * y);

	template <typename U, typename UTYPE>
	friend void csc_gaxpy_trans (const Csc<U, UTYPE> & A, U * x, U * y);
	
	template <int D, typename NT, typename IT>
	friend void csc_gaxpy_mm(const Csc<NT,IT> & A, array<NT,D> * x, array<NT,D> * y);
	
	template <int D, typename NT, typename IT>
	friend void csc_gaxpy_mm_trans(const Csc<NT,IT> & A, array<NT,D> * x, array<NT,D> * y);
};

/* y = A*x+y */
template <typename T, typename ITYPE>
void csc_gaxpy (const Csc<T, ITYPE> & A, T * x, T * y)
{
    if(A.issym)
    {
    	for (ITYPE j = 0 ; j < A.n ; ++j)	// for all columns of A
    	{
			for (ITYPE k = A.jc [j] ; k < A.jc [j+1] ; ++k)	
			{
				y [ A.ir[k] ] += A.num[k]  * x [j] ;
				if( j != A.ir[k] )
					y [ j ] += A.num[k] * x[ A.ir[k] ] ;	// perform the symmetric update
			}
		}
    }
    else
    {
    	for (ITYPE j = 0 ; j < A.n ; ++j)	// for all columns of A
    	{
			for (ITYPE k = A.jc [j] ; k < A.jc [j+1] ; ++k)	// scale jth column with x[j]
			{
				y [ A.ir[k] ] += A.num[k]  * x [j] ;
			}
		}
    }
}


/* y = A' x + y */
template <typename T, typename ITYPE>
void csc_gaxpy_trans(const Csc<T,ITYPE> & A, T * x, T * y)
{
	if(A.issym)
	{
		cout << "Trying to run A'x on a symmetric matrix doesn't make sense" << endl;
		cout << "Are you sure you're using the right data structure?" << endl;
		return;	
	}
	
	for (ITYPE j = 0; j< A.n; ++j)
	{
		for(ITYPE k= A.jc[j]; k < A.jc[j+1]; ++k)
		{
			y[j] += A.num[k] * x [ A.ir[k] ]; 
		}
	}
}


/* Y = A X + Y */
template <int D, typename NT, typename IT>
void csc_gaxpy_mm(const Csc<NT,IT> & A, array<NT,D> * x, array<NT,D> * y)
{
	if(A.issym)
	{
		cout << "Symmetric csc_gaxpy_mm not implemented yet" << endl;
		return;	
	}
	
	for (IT j = 0 ; j < A.n ; ++j)	// for all columns of A
	{
		for (IT k = A.jc[j] ; k < A.jc[j+1] ; ++k)	// scale jth column with x[j]
		{
			for(int i=0; i<D; ++i)
			{
				y[A.ir[k]][i] += A.num[k] * x[j][i];
			}
		}
	}
	
}


/* Y = A' X + Y */
template <int D, typename NT, typename IT>
void csc_gaxpy_mm_trans(const Csc<NT,IT> & A, array<NT,D> * x, array<NT,D> * y)
{
	if(A.issym)
	{
		cout << "Trying to run A'x on a symmetric matrix doesn't make sense" << endl;
		cout << "Are you sure you're using the right data structure?" << endl;
		return;	
	}
	
	for (IT j = 0; j< A.n; ++j)
	{
		for(IT k= A.jc[j]; k < A.jc[j+1]; ++k)
		{
			for(int i=0; i<D; ++i)
			{
				y[j][i] +=  A.num[k] * x[A.ir[k]][i];
			}	
		}
	}
}


#include "csc.cpp"	// Template member function definitions need to be known to the compiler
#endif

