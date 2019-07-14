#include "spvec.h"
#include "utility.h"
#if (__GNUC__ == 4 && (__GNUC_MINOR__ < 7) )
	#include "randgen.h"
#else
	#include <random>
#endif
#include <cassert>

// constructor that generates a junk dense vector 
template <class T, class ITYPE>
Spvec<T,ITYPE>::Spvec (ITYPE dim)
{
	assert(dim != 0);
	n = static_cast<ITYPE>(ceil(static_cast<float>(dim)/RBDIM)) * RBDIM;
	padding = n-dim;
	if(padding)	
		cout << "Padded vector to size " << n << " for register blocking" << endl; 
	arr = new T[n];
}

template <class T, class ITYPE>
Spvec<T,ITYPE>::Spvec (T * darr, ITYPE dim)
{
	assert(dim != 0);

	n = static_cast<ITYPE>(ceil(static_cast<float>(dim)/RBDIM)) * RBDIM;
	padding = n-dim;
	if(padding)
		cout << "Padded vector to size " << n << " for register blocking" << endl; 

	arr = new T[n]();  // zero initialized PID

	for(ITYPE i=0; i< n; ++i)
	{
		arr[i] = darr[i];
	}
}

// copy constructor
template <class T, class ITYPE>
Spvec<T,ITYPE>::Spvec (const Spvec<T, ITYPE> & rhs): n(rhs.n),padding(rhs.padding)
{
	if(n > 0)
	{
		arr = new T[n];

		for(ITYPE i=0; i< n; ++i)		
			arr[i]= rhs.arr[i];
	}
}

template <class T, class ITYPE>
Spvec<T,ITYPE> & Spvec<T,ITYPE>::operator= (const Spvec<T,ITYPE> & rhs)
{
	if(this != &rhs)		
	{
		if(n > 0)
		{
			delete [] arr;
		}

		n	= rhs.n;
		padding = rhs.padding;
		if(n > 0)
		{
			arr = new T[n];
			for(ITYPE i=0; i< n; ++i)		
				arr[i]= rhs.arr[i];
		}
	}
	return *this;
}


template <class T, class ITYPE>
Spvec<T,ITYPE>::~Spvec()
{
	if ( n > 0)
	{
		delete [] arr;
	}
}

template <class T, class ITYPE>
Spvec<T,ITYPE> & Spvec<T,ITYPE>::operator+=(const Matmul< Csc<T, ITYPE>, Spvec<T,ITYPE> > & matmul)
{
	if((n-padding == matmul.op1.rowsize()) && (matmul.op1.colsize() == matmul.op2.size()))		// check compliance
	{
		csc_gaxpy(matmul.op1, const_cast< T * >(matmul.op2.arr), arr); 
	}
	else
	{
		cout<< "Detected noncompliant matvec..." << endl;
	}
	return *this;
}

template <class T, class ITYPE>
Spvec<T,ITYPE> & Spvec<T,ITYPE>::operator+=(const Matmul< BiCsb<T, ITYPE>, Spvec<T,ITYPE> > & matmul)
{
	typedef PTSR< T, T> PTDD;
	if((n-padding == matmul.op1.rowsize()) && (matmul.op1.colsize() == matmul.op2.size()))		// check compliance
	{
		bicsb_gespmv<PTDD>(matmul.op1, matmul.op2.arr, arr); 
	}
	else
	{
		cout<< "Detected noncompliant matvec..." << endl;
	}
	return *this;
}

// populate the vector with random entries
// currently, only works for T "double" and "float"
template <class T, class ITYPE>
void Spvec<T,ITYPE>::fillrandom()
{
#if (__GNUC__ == 4 && (__GNUC_MINOR__ < 7) )
	RandGen G;
	for(ITYPE i=0; i< n; ++i)
	{
		arr[i] = G.RandReal();
	}
#else
	std::uniform_real_distribution<T> distribution(0.0f, 1.0f); //Values between 0 and 1
	std::mt19937 engine; // Mersenne twister MT19937
	auto generator = std::bind(distribution, engine);
	std::generate_n(arr, n, generator); 
#endif	
}

// populate the vector with zeros
template <class T, class ITYPE>
void Spvec<T,ITYPE>::fillzero()
{
	for(ITYPE i=0; i< n; ++i)
	{
		arr[i] = 0;
	}
}

template <typename NT, typename IT>
void Verify(Spvec<NT, IT> & control, Spvec<NT, IT> & test, string name, IT m)
{
    vector<NT>error(m);
    std::transform(&control[0], (&control[0])+m, &test[0], error.begin(), absdiff<NT>());
    auto maxelement = std::max_element(error.begin(), error.end());
    cout << "Max error is: " << *maxelement << " on y[" << maxelement-error.begin()<<"]=" << test[maxelement-error.begin()] << endl;
    NT machEps = machineEpsilon<NT>();
    cout << "Absolute machine epsilon is: " << machEps <<" and y[" << maxelement-error.begin() << "]*EPSILON becomes "
    << machEps * test[maxelement-error.begin()] << endl;
    
    NT sqrtm = sqrt(static_cast<NT>(m));
    cout << "sqrt(n) * relative error is: " << abs(machEps * test[maxelement-error.begin()]) * sqrtm << endl;
    if ( (abs(machEps * test[maxelement-error.begin()]) * sqrtm) < abs(*maxelement))
    {
        cout << "*** ATTENTION ***: error is more than sqrt(n) times the relative machine epsilon" << endl;
    }
    
#ifdef DEBUG
    cout << "<index, control, test>: \n";
    for(IT i=0; i<m; ++i)
    {
        if(error[i] > abs(sqrtm * machEps * test[i]))
        {
            cout << i << "\t" << control[i] << " " << test[i] << "\n";
        }
    }
#endif
}


