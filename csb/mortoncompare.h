#ifndef _MORTONCOMPARE_H_
#define _MORTONCOMPARE_H_


template <class ITYPE>
class MortonCompare: public binary_function< ITYPE , ITYPE , bool >    // (par1, par2, return_type)
{
public:
	MortonCompare()  {}
	MortonCompare (ITYPE nrbits, ITYPE ncbits, ITYPE rmask, ITYPE cmask ) 
		: nrowbits(nrbits), ncolbits(ncbits), rowmask(rmask), colmask(cmask) {}

	// rhs is the splitter that is already in bit-interleaved order
	// lhs is the actual value that is in row-major order
   	 bool operator()(const ITYPE & lhs, const ITYPE & rhs) const
    	{
		ITYPE rlowbits = ((lhs >> ncolbits) & rowmask);
		ITYPE clowbits = (lhs & colmask);
		ITYPE bikey = BitInterleaveLow(rlowbits, clowbits);

		return bikey < rhs;
    	}

private:
  	ITYPE nrowbits;
  	ITYPE ncolbits;
  	ITYPE rowmask;
  	ITYPE colmask;
};

template <class ITYPE>
class MortCompSym: public binary_function< ITYPE , ITYPE , bool >    // (par1, par2, return_type)
{
public:
	MortCompSym()  {}
	MortCompSym(ITYPE bits, ITYPE lowmask): nbits(bits), lmask(lowmask) {}

	// rhs is the splitter that is already in bit-interleaved order
	// lhs is the actual value that is in row-major order
   	 bool operator()(const ITYPE & lhs, const ITYPE & rhs) const
    	{
		ITYPE rlowbits = ((lhs >> nbits) & lmask);
		ITYPE clowbits = (lhs & lmask);
		ITYPE bikey = BitInterleaveLow(rlowbits, clowbits);

		return bikey < rhs;
    	}

private:
  	ITYPE nbits;
  	ITYPE lmask;

};

#endif

