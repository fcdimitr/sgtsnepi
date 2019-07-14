/* author: Aydin Buluc (aydin@cs.ucsb.edu) ---------------------- */
/*  description: Helper class in order to get rid of copying */
/* 	and temporaries during the y += A*x or A+=B*C calls */
/* acknowlegment: This technique is described in Stroustrup, */
/* 	The C++ Programming Language, 3rd Edition, */
/* 	Section 22.4.7 [Temporaries, Copying and Loops] */



#ifndef _MAT_MUL_H
#define _MAT_MUL_H


template <class OPT1, class OPT2>
struct Matmul
{
	const OPT1 & op1;	// just keeps references to objects
	const OPT2 & op2;

	// Constructor 
	Matmul(const OPT1 & operand1, const OPT2 & operand2): op1(operand1), op2(operand2) { }

	// No need for operator BT() because we have the corresponding copy constructor 
	// and assignment operators to evaluate and return result !
};

template <class OPT1, class OPT2>
inline Matmul< OPT1,OPT2  > operator* (const OPT1 & operand1, const OPT2 & operand2)
{
	return Matmul< OPT1,OPT2 >(operand1,operand2);	//! Just defer the multiplication
}

#endif
	
