#include "csbsym.h"
#include <iterator>
#include "utility.h"

// Choose block size as big as possible given the following constraints
// 1) The bot array is addressible by IT
// 2) The parts of x & y vectors that a block touches fits into L2 cache [assuming a saxpy() operation]
// 3) There's enough parallel slackness for block rows (at least SLACKNESS * CILK_NPROC)
template <class NT, class IT>
void CsbSym<NT, IT>::Init(int workers, IT forcelogbeta)
{
	ispar = (workers > 1);
	IT roundup = nextpoweroftwo(n);

	// if indices are negative, highestbitset returns -1, 
	// but that will be caught by the sizereq below
	IT nbits = highestbitset(roundup);
	bool sizereq;
	if (ispar)
	{
		sizereq = (IntPower<2>(nbits) > SLACKNESS * workers);
	}
	else
	{
		sizereq = (nbits > 1);
	}
	if(!sizereq)
	{
		cerr << "Matrix too small for this library" << endl;
		return;
	}

	nlowbits = nbits-1;	
	IT inf = numeric_limits<IT>::max();
	IT maxbits = highestbitset(inf);

	nhighbits = nbits-nlowbits;	// # higher order bits for rows (has at least one bit)
	if(ispar)
	{
		while(IntPower<2>(nhighbits) < SLACKNESS * workers)
		{
			nhighbits++;
			nlowbits--;
		}
	}

	// calculate the space that suby and subx occupy in L2 cache
	IT yL2 = IntPower<2>(nlowbits) * sizeof(NT);
	while(yL2 > L2SIZE)
	{
		yL2 /= 2;
		nhighbits++;
		nlowbits--;
	}

	lowmask = IntPower<2>(nlowbits) - 1;
	if(forcelogbeta != 0)
	{
		IT candlowmask  = IntPower<2>(forcelogbeta) -1;
		cout << "Forcing beta to "<< (candlowmask+1) << " instead of the chosen " << (lowmask+1) << endl;
		cout << "Warning : No checks are performed on the beta you have forced, anything can happen !" << endl;
		lowmask = candlowmask;
		nlowbits = forcelogbeta;
		nhighbits = nbits-nlowbits; 
	}
	else 
	{
		double sqrtn = sqrt(static_cast<double>(n));
		IT logbeta = static_cast<IT>(ceil(log2(sqrtn))) + 2;
		if(nlowbits > logbeta)
		{
			nlowbits = logbeta;
			lowmask = IntPower<2>(logbeta) -1;
			nhighbits = nbits-nlowbits;
		}
		cout << "Beta chosen to be "<< (lowmask+1) << endl;
	}
	highmask = ((roundup - 1) ^ lowmask);
	
	IT blcdim = lowmask + 1;
        ncsb = static_cast<IT>(ceil(static_cast<double>(n) / static_cast<double>(blcdim)));
	
	blcrange = (lowmask+1) * (lowmask+1);	// range indexed by one block
	mortoncmp = MortCompSym<IT>(nlowbits, lowmask);
}


// copy constructor
template <class NT, class IT>
CsbSym<NT, IT>::CsbSym (const CsbSym<NT,IT> & rhs)
: nz(rhs.nz), n(rhs.n), blcrange(rhs.blcrange), ncsb(rhs.ncsb), nhighbits(rhs.nhighbits), nlowbits(rhs.nlowbits), 
highmask(rhs.highmask), lowmask(rhs.lowmask), mortoncmp(rhs.mortoncmp), ispar(rhs.ispar), diagonal(rhs.diagonal)
{
	if(nz > 0)  // nz > 0 iff nrb > 0
	{
		num = new NT[nz]();
		bot = new IT[nz];

		copy ( rhs.num, rhs.num+nz, num);
		copy ( rhs.bot, rhs.bot+nz, bot );
	}
	if ( ncsb > 0)
	{
		top = new IT* [ncsb];
		for(IT i=0; i<ncsb; ++i)
			top[i] = new IT[ncsb-i+1]; 
		for(IT i=0; i<ncsb; ++i)
			for(IT j=0; j <= (ncsb-i); ++j) 
				top[i][j] = rhs.top[i][j];
	}
}

template <class NT, class IT>
CsbSym<NT, IT> & CsbSym<NT, IT>::operator= (const CsbSym<NT, IT> & rhs)
{
	if(this != &rhs)		
	{
		if(nz > 0)	// if the existing object is not empty
		{
			// make it empty
			delete [] bot;
			delete [] num;
		}
		if(ncsb > 0)
		{
			for(IT i=0; i<ncsb; ++i)
				delete [] top[i];
			delete [] top;
		}
		ispar 	= rhs.ispar;
		nz	= rhs.nz;
		n	= rhs.n;
		ncsb 	= rhs.ncsb;
		blcrange = rhs.blcrange;
		mortoncmp = rhs.mortoncmp;
		
		nhighbits = rhs.nhighbits;
		nlowbits = rhs.nlowbits;
		highmask = rhs.highmask;
		lowmask	 = rhs.lowmask;
		diagonal = rhs.diagonal;	// copy the whole sparse vector

		if(nz > 0)	// if the copied object is not empty
		{
			num = new NT[nz]();
			bot = new IT[nz];

			copy ( rhs.num, rhs.num+nz, num);
			copy ( rhs.bot, rhs.bot+nz, bot );
		}
		if(ncsb > 0)
		{
			top = new IT* [ncsb];
			for(IT i=0; i<ncsb; ++i)
				top[i] = new IT[ncsb-i+1]; 
			for(IT i=0; i<ncsb; ++i)
				for(IT j=0; j <= (ncsb-i); ++j) 
					top[i][j] = rhs.top[i][j];
		}
	}
	return *this;
}

template <class NT, class IT>
CsbSym<NT, IT>::~CsbSym()
{
	if( nz > 0)
	{
		delete [] bot;
		delete [] num;
	}
	if ( ncsb > 0)
	{
		for(IT i=0; i<ncsb; ++i)
			delete [] top[i];
		delete [] top;
	}
}

template <class NT, class IT>
CsbSym<NT, IT>::CsbSym (Csc<NT, IT> & csc, int workers):nz(csc.nz), n(csc.n)
{
        typedef std::pair<IT, IT> ipair;
        typedef std::pair<IT, ipair> mypair;

        assert(nz != 0 && n != 0);
        Init(workers);

	top = new IT* [ncsb];
	for(IT i=0; i<ncsb; ++i)
		top[i] = new IT[ncsb-i+1]; 

        mypair * pairarray = new mypair[nz];
        IT k = 0;
        for(IT j = 0; j < n; ++j)
        {
                for (IT i = csc.jc [j] ; i < csc.jc[j+1] ; ++i) // scan the jth column
                {
                        // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
                        IT hindex = (((highmask &  csc.ir[i] ) >> nlowbits) << nhighbits) | ((highmask & j) >> nlowbits);
                        IT lindex = ((lowmask &  csc.ir[i]) << nlowbits) | (lowmask & j) ;

                        // i => location of that nonzero in csc.ir and csc.num arrays
                        pairarray[k++] = mypair(hindex, ipair(lindex,i));
                }
        }
        sort(pairarray, pairarray+nz);  // sort according to hindex
        SortBlocks(pairarray, csc.num);
        delete [] pairarray;
}

template <class NT, class IT>
void CsbSym<NT, IT>::SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val)
{
        typedef pair<IT, pair<IT, IT> > mypair;
        IT cnz = 0;
	vector<IT> tempbot;
	vector<NT> tempnum;
        IT ldim = IntPower<2>(nhighbits);  // leading dimension (not always equal to ncsb)
        for(IT i = 0; i < ncsb; ++i)
        {
                for(IT j = 0; j < (ncsb-i); ++j)
                {
                        top[i][j] = tempbot.size();
                        IT prevcnz = cnz;
                        std::vector<mypair> blocknz;
                        while(cnz < nz && pairarray[cnz].first == ((i*ldim)+(j+i)) )        // as long as we're in this block
                        {
                                IT interlowbits = pairarray[cnz].second.first;
                                IT rlowbits = ((interlowbits >> nlowbits) & lowmask);
                                IT clowbits = (interlowbits & lowmask);
                                IT bikey = BitInterleaveLow(rlowbits, clowbits);

				if(j == 0 && rlowbits == clowbits)
				{
					diagonal.push_back(make_pair((i << nlowbits)+rlowbits, val[pairarray[cnz++].second.second]));
				}
				else
				{
                                	blocknz.push_back(mypair(bikey, pairarray[cnz++].second));
				}
                        }
                        // sort the block into bitinterleaved order
                        sort(blocknz.begin(), blocknz.end());
			typename vector<mypair>::iterator itr; 
	
			for( itr = blocknz.begin(); itr != blocknz.end(); ++itr) 
                        {
                                tempbot.push_back( itr->second.first );
                                tempnum.push_back( val[itr->second.second] );
                        }
                }
                top[i][ncsb-i] = tempbot.size();
        }

	assert (cnz == (tempbot.size() + diagonal.size()));
	nz = tempbot.size();	// update the number of off-diagonal nonzeros
	bot = new IT[nz];
	num = new NT[nz];

	copy(tempbot.begin(), tempbot.end(), bot);
	copy(tempnum.begin(), tempnum.end(), num);
	sort(diagonal.begin(), diagonal.end());
}

template<class NT, class IT>
void CsbSym<NT, IT>::DivideIterationSpace(IT * & lspace, IT * & rspace, IT & lsize, IT & rsize, IT size, IT d) const
{
	if(d == 1)
	{
		lsize = size-size/2;
		rsize = size/2;
		lspace = new IT[lsize];
		rspace = new IT[rsize]; 
		for(IT i=0; i<rsize; ++i)	// we alternate indices
		{
			lspace[i] = 2*i;
			rspace[i] = 2*i+1;
		}
		if(lsize > rsize)
		{
			lspace[lsize-1] = size-1;
		}
	}
	else 
	{
		IT chunks = size / (2*d);
		int rest = size - (2*d*chunks);	// rest is modulus 2d
		lsize = d*chunks;	// initial estimates
		rsize = d*chunks; 
		if(rest > d)	// first d goes to lsize, rest goes to rsize
		{
			rsize += (rest-d);
			lsize += d;
		}
		else	// all goes to lsize
		{
			lsize += rest;
		}
		lspace = new IT[lsize];
		rspace = new IT[rsize];
		int remrest = (int) rest;	// needs to be signed integer since we're looping it until negative
		if(d == 2)
		{
			for(IT i=0; i<chunks; ++i)	// we alternate indices
			{
				lspace[2*i+0] = 4*i+0;
				lspace[2*i+1] = 4*i+1;
				rspace[2*i+0] = 4*i+2;
				rspace[2*i+1] = 4*i+3; 
			}
			if(remrest-- > 0)	lspace[2*chunks+0] = 4*chunks+0;		
			if(remrest-- > 0)	lspace[2*chunks+1] = 4*chunks+1;
			if(remrest-- > 0)	rspace[2*chunks+0] = 4*chunks+2;
		}
		else if(d == 3)
		{
			for(IT i=0; i<chunks; ++i)
			{
				lspace[3*i+0] = 6*i+0;
				lspace[3*i+1] = 6*i+1;
				lspace[3*i+2] = 6*i+2;
				rspace[3*i+0] = 6*i+3;
				rspace[3*i+1] = 6*i+4;
				rspace[3*i+2] = 6*i+5;
			}
			if(remrest-- > 0)	lspace[3*chunks+0] = 6*chunks+0;
			if(remrest-- > 0)	lspace[3*chunks+1] = 6*chunks+1;
			if(remrest-- > 0)	lspace[3*chunks+2] = 6*chunks+2;
			if(remrest-- > 0)	rspace[3*chunks+0] = 6*chunks+3;
			if(remrest-- > 0)	rspace[3*chunks+1] = 6*chunks+4;
		}		
		else if(d == 4)
		{	
			
			for(IT i=0; i<chunks; ++i)
			{
				lspace[4*i+0] = 8*i+0;
				lspace[4*i+1] = 8*i+1;
				lspace[4*i+2] = 8*i+2;
				lspace[4*i+3] = 8*i+3;
				rspace[4*i+0] = 8*i+4;
				rspace[4*i+1] = 8*i+5;
				rspace[4*i+2] = 8*i+6;
				rspace[4*i+3] = 8*i+7;
			}
			if(remrest-- > 0)	lspace[4*chunks+0] = 8*chunks+0;
			if(remrest-- > 0)	lspace[4*chunks+1] = 8*chunks+1;
			if(remrest-- > 0)	lspace[4*chunks+2] = 8*chunks+2;
			if(remrest-- > 0)	lspace[4*chunks+3] = 8*chunks+3;
			if(remrest-- > 0)	rspace[4*chunks+0] = 8*chunks+4;
			if(remrest-- > 0)	rspace[4*chunks+1] = 8*chunks+5;
			if(remrest-- > 0)	rspace[4*chunks+2] = 8*chunks+6;
		}	
		else
		{
			cout << "Diagonal d = " << d << " is not yet supported" << endl;
		}	
	}
}

template<class NT, class IT>
void CsbSym<NT, IT>::MultAddAtomics(NT * __restrict y, const NT * __restrict x, const IT d) const
{
	cilk_for(IT i=0; i< ncsb-d; ++i)	// all blocks at the dth diagonal and beyond
	{
		IT rhi = (i << nlowbits);
                NT * __restrict suby = &y[rhi];
		const NT * __restrict subx_mirror = &x[rhi];
		
		cilk_for(IT j=d; j < (ncsb-i); ++j)
		{
			IT chi = ((j+i) << nlowbits);
			const NT * __restrict subx = &x[chi];
			NT * __restrict suby_mirror = &y[chi];

			IT * __restrict r_bot = bot;
        		NT * __restrict r_num = num;
			for(IT k=top[i][j]; k<top[i][j+1]; ++k)
			{
				IT ind = r_bot[k];
				NT val = r_num[k];

				IT rli = ((r_bot[k] >> nlowbits) & lowmask);
				IT cli = (r_bot[k] & lowmask);

				atomicallyIncrementDouble(&suby[rli], val * subx[cli]);
				atomicallyIncrementDouble(&suby_mirror[cli], val * subx_mirror[rli]);
#ifdef STATS
			        atomicflops += 2;
#endif
			}
		}
	}
}


template <class NT, class IT>
void CsbSym<NT, IT>::MultMainDiag(NT * __restrict y, const NT * __restrict x) const
{
	if(Imbalance(0) > 2 * BALANCETH)
	{
		cilk_for(IT i=0; i< ncsb; ++i)	// in main diagonal, j = i
		{
			IT hi = (i << nlowbits);
                	NT * __restrict suby = &y[hi];
			const NT * __restrict subx = &x[hi];
	
			if(i == (ncsb-1) && (n-hi) <= lowmask)	// last iteration and it's irregular (can't parallelize)
			{
				IT * __restrict r_bot = bot;
       		 		NT * __restrict r_num = num;
				for(IT k=top[i][0]; k<top[i][1]; ++k)
				{
					IT ind = r_bot[k];
					NT val = r_num[k];

					IT rli = ((ind >> nlowbits) & lowmask);
					IT cli = (ind & lowmask);
					
					suby[rli] += val * subx[cli];
					suby[cli] += val * subx[rli];	// symmetric update
				}
			}
			else
			{
				BlockTriPar(top[i][0], top[i][1], subx, suby, 0, blcrange, BREAKEVEN * (nlowbits+1));
			}
		}
	}
	else	// No need for block parallelization
	{
		cilk_for(IT i=0; i< ncsb; ++i)	// in main diagonal, j = i
		{
			IT hi = (i << nlowbits);
                	NT * __restrict suby = &y[hi];
			const NT * __restrict subx = &x[hi];
	
			IT * __restrict r_bot = bot;
       		 	NT * __restrict r_num = num;
			for(IT k=top[i][0]; k<top[i][1]; ++k)
			{
				IT ind = r_bot[k];
				NT val = r_num[k];
		
				IT rli = ((ind >> nlowbits) & lowmask);
				IT cli = (ind & lowmask);

				suby[rli] += val * subx[cli];
				suby[cli] += val * subx[rli];	// symmetric update
			}
		}
	}
	const IT diagsize = diagonal.size();
	cilk_for(IT i=0; i < diagsize; ++i)
	{
		y[diagonal[i].first] += diagonal[i].second * x[diagonal[i].first];	// process the diagonal
	}
}


// Multiply the dth block diagonal
// which is composed of blocks A[i][i+n]
template <class NT, class IT>
void CsbSym<NT, IT>::MultDiag(NT * __restrict y, const NT * __restrict x, const IT d) const
{
	if(d == 0)
	{
		MultMainDiag(y, x);
		return;
	}
	IT * lspace;
	IT * rspace;
	IT lsize, rsize;
	DivideIterationSpace(lspace, rspace, lsize, rsize, ncsb-d, d);
	IT lsum = 0;
	IT rsum = 0;
	for(IT k=0; k<lsize; ++k)
	{
		lsum += top[lspace[k]][d+1] - top[lspace[k]][d];
	}
	for(IT k=0; k<rsize; ++k)
	{
		rsum += top[rspace[k]][d+1] - top[rspace[k]][d];
	}
	float lave = lsum / lsize;
	float rave = rsum / rsize;

	cilk_for(IT i=0; i< lsize; ++i)	// in the dth diagonal, j = i+d
	{
		IT rhi = (lspace[i] << nlowbits) ;
		IT chi = ((lspace[i]+d) << nlowbits);
               	NT * __restrict suby = &y[rhi];
                NT * __restrict suby_mirror = &y[chi];
		const NT * __restrict subx = &x[chi];
		const NT * __restrict subx_mirror = &x[rhi];
		
		if((top[lspace[i]][d+1] - top[lspace[i]][d] > BALANCETH * lave)	// relative denser block
			&& (!(lspace[i] == (ncsb-d-1) && (n-chi) <= lowmask)))	// and parallelizable
		{
			BlockPar(top[lspace[i]][d], top[lspace[i]][d+1], subx, subx_mirror, suby, suby_mirror, 0, blcrange, BREAKEVEN * (nlowbits+1));
		}
		else
		{	
			IT * __restrict r_bot = bot;
	       	 	NT * __restrict r_num = num;
			for(IT k=top[lspace[i]][d]; k<top[lspace[i]][d+1]; ++k)
			{
				IT rli = ((r_bot[k] >> nlowbits) & lowmask);
				IT cli = (r_bot[k] & lowmask);
				suby[rli] += r_num[k] * subx[cli];
				suby_mirror[cli] += r_num[k] * subx_mirror[rli];	// symmetric update
			}
		}
	}
	
	cilk_for(IT j=0; j< rsize; ++j)
	{
		IT rhi = (rspace[j] << nlowbits) ;
		IT chi = ((rspace[j]+d) << nlowbits);
       		NT * __restrict suby = &y[rhi];
                NT * __restrict suby_mirror = &y[chi];
		const NT * __restrict subx = &x[chi];
		const NT * __restrict subx_mirror = &x[rhi];

		if((top[rspace[j]][d+1] - top[rspace[j]][d] > BALANCETH * rave)	// relative denser block
			&& (!(rspace[j] == (ncsb-d-1) && (n-chi) <= lowmask))) // and parallelizable
		{
			BlockPar(top[rspace[j]][d], top[rspace[j]][d+1], subx, subx_mirror, suby, suby_mirror, 0, blcrange, BREAKEVEN * (nlowbits+1));
		}
		else
		{
			IT * __restrict r_bot = bot;
        		NT * __restrict r_num = num;
			for(IT k=top[rspace[j]][d]; k<top[rspace[j]][d+1]; ++k)
			{
				IT rli = ((r_bot[k] >> nlowbits) & lowmask);
				IT cli = (r_bot[k] & lowmask);
				suby[rli] += r_num[k] * subx[cli];
				suby_mirror[cli] += r_num[k] * subx_mirror[rli];	// symmetric update
			}
		}
	}
	delete [] lspace;
	delete [] rspace;
}

// Block parallelization for upper triangular compressed sparse blocks
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the `same block
// PRECONDITION: rangeend-rangebeg is a power of two 
template <class NT, class IT>
void CsbSym<NT, IT>::BlockTriPar(IT start, IT end, const NT * __restrict subx, NT * __restrict suby, 
				IT rangebeg, IT rangeend, IT cutoff) const
{
	assert(IsPower2(rangeend-rangebeg));
	if(end - start < cutoff)
	{
		IT * __restrict r_bot = bot;
        	NT * __restrict r_num = num;
		for(IT k=start; k<end; ++k)
		{
			IT ind = r_bot[k];
			NT val = r_num[k];
			
			IT rli = ((ind >> nlowbits) & lowmask);
			IT cli = (ind & lowmask);

			suby[rli] += val * subx[cli];
			suby[cli] += val * subx[rli];	// symmetric update
		}
	}
	else
	{
		// Lower_bound is a version of binary search: it attempts to find the element value in an ordered range [first, last) 
		// Specifically, it returns the first position where value could be inserted without violating the ordering
		IT halfrange = (rangebeg+rangeend)/2;
		IT qrt1range = (rangebeg+halfrange)/2;
		IT qrt3range = (halfrange+rangeend)/2;

		IT * mid = std::lower_bound(&bot[start], &bot[end], halfrange, mortoncmp);	// divides in mid column
		IT * right = std::lower_bound(mid, &bot[end], qrt3range, mortoncmp);

		/* -------
		   | 0 2 |
		   | 1 3 |
		   ------- */
		// subtracting two pointers pointing to the same array gives you the # of elements separating them
		// In the symmetric case, quadrant "1" doesn't exist (size1 = 0)
		IT size0 = static_cast<IT> (mid - &bot[start]);
		IT size2 = static_cast<IT> (right - mid);
		IT size3 = static_cast<IT> (&bot[end] - right);

		IT ncutoff = std::max<IT>(cutoff/2, MINNNZTOPAR);
	    
		cilk_spawn BlockTriPar(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
		BlockTriPar(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);			// multiply subblock_3
		cilk_sync;

		BlockPar(start+size0, end-size3, subx, subx, suby, suby, halfrange, qrt3range, ncutoff); // multiply subblock_2
	}
}

// Parallelize the block itself
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the same block
// PRECONDITION: rangeend-rangebeg is a power of two 
// TODO: we rely on the particular implementation of lower_bound for correctness, which is dangerous !
//		 what if lhs (instead of rhs) parameter to the comparison object is the splitter?
template <class NT, class IT>
void CsbSym<NT, IT>::BlockPar(IT start, IT end, const NT * __restrict subx, const NT * __restrict subx_mirror, 
			NT * __restrict suby, NT * __restrict suby_mirror, IT rangebeg, IT rangeend, IT cutoff) const
{
	assert(IsPower2(rangeend-rangebeg));
	if(end - start < cutoff)
	{
		IT * __restrict r_bot = bot;
        	NT * __restrict r_num = num;
		for(IT k=start; k<end; ++k)
		{
			IT ind = r_bot[k];
			NT val = r_num[k];
					
			IT rli = ((ind >> nlowbits) & lowmask);
			IT cli = (ind & lowmask);

			suby[rli] += val * subx[cli];
			suby_mirror[cli] += val * subx_mirror[rli];	// symmetric update
		}
	}
	else
	{
		// Lower_bound is a version of binary search: it attempts to find the element value in an ordered range [first, last) 
		// Specifically, it returns the first position where value could be inserted without violating the ordering
		IT halfrange = (rangebeg+rangeend)/2;
		IT qrt1range = (rangebeg+halfrange)/2;
		IT qrt3range = (halfrange+rangeend)/2;

		IT * mid = std::lower_bound(&bot[start], &bot[end], halfrange, mortoncmp);
		IT * left = std::lower_bound(&bot[start], mid, qrt1range, mortoncmp);
		IT * right = std::lower_bound(mid, &bot[end], qrt3range, mortoncmp);

		/* -------
		   | 0 2 |
		   | 1 3 |
		   ------- */
		// subtracting two pointers pointing to the same array gives you the # of elements separating them
		// we're *sure* that the differences are 1) non-negative, 2) small enough to be indexed by an IT
		IT size0 = static_cast<IT> (left - &bot[start]);
		IT size1 = static_cast<IT> (mid - left);
		IT size2 = static_cast<IT> (right - mid);
		IT size3 = static_cast<IT> (&bot[end] - right);

		IT ncutoff = std::max<IT>(cutoff/2, MINNNZTOPAR);
	    
		// We only perform [0,3] in parallel and then [1,2] in parallel because the symmetric update causes races when 
		// performing [0,1] in parallel (as it would perform [0,2] in the fictitious lower triangular part)
		cilk_spawn BlockPar(start, start+size0, subx, subx_mirror, suby, suby_mirror, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
		BlockPar(end-size3, end, subx, subx_mirror, suby, suby_mirror, qrt3range, rangeend, ncutoff);			// multiply subblock_3
		cilk_sync;

		cilk_spawn BlockPar(start+size0, start+size0+size1, subx, subx_mirror, suby, suby_mirror, qrt1range, halfrange, ncutoff);	// multiply subblock_1
		BlockPar(start+size0+size1, end-size3, subx, subx_mirror, suby, suby_mirror, halfrange, qrt3range, ncutoff);		// multiply subblock_2
		cilk_sync;
	}
}


// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
void CsbSym<NT, IT>::SeqSpMV(const NT * __restrict x, NT * __restrict y) const
{
	const IT diagsize = diagonal.size();
	for(IT i=0; i < diagsize; ++i)
	{
		y[diagonal[i].first] += diagonal[i].second * x[diagonal[i].first];	// process the diagonal
	}
	for (IT i = 0 ; i < ncsb ; ++i)    // for all block rows of A 
	{
		IT rhi = (i << nlowbits);
		NT * suby = &y[rhi];
		const NT * subx_mirror = &x[rhi];

		IT * __restrict r_bot = bot;
        	NT * __restrict r_num = num;
		for (IT j = 0 ; j < (ncsb-i) ; ++j)		// for all blocks inside that block row
		{
                	// get higher order bits for column indices
                	IT chi = ((j+i) << nlowbits);
                	const NT * __restrict subx = &x[chi];
			NT * __restrict suby_mirror = &y[chi];

			for(IT k=top[i][j]; k<top[i][j+1]; ++k)
			{
				IT rli = ((r_bot[k] >> nlowbits) & lowmask);
				IT cli = (r_bot[k] & lowmask);
				NT val = r_num[k];
				suby[rli] += val * subx[cli];
				suby_mirror[cli] += val * subx_mirror[rli];	// symmetric update
			}
		}
	}
}

// Imbalance in the dth block diagonal (the main diagonal is the 0th) 
template <class NT, class IT>
float CsbSym<NT, IT>::Imbalance(IT d) const
{
	if (ncsb <= d+1)
	{
		return 0.0; //pointless
	}

	IT size = ncsb-d-1;
	IT * sums = new IT[size];
        for(size_t i=0; i< size; ++i)
        {
		sums[i] = top[i][d+1] - top[i][d]; 
        }
	IT max = *max_element(sums, sums+size);
	IT mean = accumulate(sums, sums+size, 0.0) / size;
	delete [] sums;

        return static_cast<float>(max) / mean;
}


// Total number of nonzeros in the dth block diagonal (the main diagonal is the 0th) 
template <class NT, class IT>
IT CsbSym<NT, IT>::nzsum(IT d) const
{
        IT sum = 0; 
	for(size_t i=0; i< ncsb-d; ++i)
        {
		sum += (top[i][d+1] - top[i][d]); 
        }
        return sum;
}

// Print stats to an ofstream object
template <class NT, class IT>
ofstream & CsbSym<NT, IT>::PrintStats(ofstream & outfile) const 
{
	if(nz == 0)
	{
		outfile << "## Matrix Doesn't have any nonzeros" <<endl;
		return outfile;
	}
	const IT ntop = ncsb * ncsb; 	

	outfile << "## Average block is of dimensions "<< lowmask+1 << "-by-" << lowmask+1 << endl;
	outfile << "## Number of real blocks is "<< ntop << endl;
	outfile << "## Main (0th) block diagonal imbalance: " << Imbalance(0) << endl;
	outfile << "## 1st block diagonal imbalance: " << Imbalance(1) << endl;
	outfile << "## 2nd block diagonal imbalance: " << Imbalance(2) << endl;

	outfile << "## nnz ratios (block diagonal 0,1,2): " << static_cast<float>(nzsum(0)) / nz << ", " 
		<< static_cast<float>(nzsum(1)) / nz << ", " << static_cast<float>(nzsum(2)) / nz << endl; 
	outfile << "## atomics ratio: " << static_cast<float>(nz-nzsum(0)-nzsum(1)-nzsum(2))/nz << endl;
	
	std::vector<int> blocksizes;
	for(IT i=0; i<ncsb; ++i)
	{
		for(IT j=0; j < (ncsb-i); ++j) 
		{
			blocksizes.push_back(static_cast<int> (top[i][j+1]-top[i][j]));
		}
	}	
	sort(blocksizes.begin(), blocksizes.end());
	outfile<< "## Total number of nonzeros: " << 2*nz +diagonal.size()<< endl;
	outfile<< "## Total number of stored nonzeros: "<< nz+diagonal.size() << endl;
	outfile<< "## Size of diagonal: " << diagonal.size() << endl;

	outfile << "## Nonzero distribution (sorted) of blocks follows: \n" ;
	std::copy(blocksizes.begin(), blocksizes.end(), ostream_iterator<int>(outfile,"\n"));
	outfile << endl;
	return outfile;
}


template <class NT, class IT>
ofstream & CsbSym<NT, IT>::Dump(ofstream & outfile) const
{
	for(typename vector< pair<IT,NT> >::const_iterator itr = diagonal.begin(); itr != diagonal.end(); ++itr)
	{	
		outfile << itr->first << " " << itr->second << "\n";
	}
	for(IT i =0; i<ncsb; ++i)
	{
		for(IT j=0; j< (ncsb-i); ++j)
		{
			outfile << "Dumping A.top(" << i << "," << j << ")" << endl;
			for(IT k=top[i][j]; k< top[i][j+1]; ++k)
			{
				IT rli = ((bot[k] >> nlowbits) & lowmask);
				IT cli = bot[k] & lowmask;
				outfile << "A(" << rli << "," << cli << ")=" << num[k] << endl;
			}
		}
	}
	return outfile;
}
