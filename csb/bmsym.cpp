#include "bmsym.h"
#include "utility.h"

// Choose block size as big as possible given the following constraints
// 1) The bot array is addressible by IT
// 2) The parts of x & y vectors that a block touches fits into L2 cache [assuming a saxpy() operation]
// 3) There's enough parallel slackness for block rows (at least SLACKNESS * CILK_NPROC)
template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::Init(int workers, IT forcelogbeta)
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
		double sqrtn = sqrt(static_cast<double>(n));	IT logbeta = static_cast<IT>(ceil(log2(sqrtn))) + 2;
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
template <class NT, class IT, unsigned TTDIM>
BmSym<NT, IT, TTDIM>::BmSym (const BmSym<NT,IT, TTDIM> & rhs)
: nz(rhs.nz), n(rhs.n), blcrange(rhs.blcrange), ncsb(rhs.ncsb), nrb(rhs.nrb), 
nhighbits(rhs.nhighbits), nlowbits(rhs.nlowbits), diagonal(rhs.diagonal),
highmask(rhs.highmask), lowmask(rhs.lowmask), mortoncmp(rhs.mortoncmp), ispar(rhs.ispar)
{
	if(nz > 0)  // nz > 0 iff nrb > 0
	{
		num = new NT[nz+2]();	// pad from both sides
		num++;

		bot = new IT[nrb];
		masks = new MTYPE[nrb];
		scansum = new IT[nrb];

		copy ( rhs.num, rhs.num+nz+1, num);
		copy ( rhs.bot, rhs.bot+nrb, bot );
		copy ( rhs.masks, rhs.masks+nrb, masks );
		copy ( rhs.scansum, rhs.scansum+nrb, scansum );
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

template <class NT, class IT, unsigned TTDIM>
BmSym<NT, IT, TTDIM> & BmSym<NT, IT, TTDIM>::operator= (const BmSym<NT, IT,TTDIM> & rhs)
{
	if(this != &rhs)		
	{
		if(nz > 0)	// if the existing object is not empty
		{
			// make it empty
			delete [] scansum;
			delete [] masks;
			delete [] bot;
			delete [] (--num);
		}
		if(ncsb > 0)
		{
			for(IT i=0; i<ncsb; ++i)
				delete [] top[i];
			delete [] top;
		}
		ispar 	= rhs.ispar;
		nz	= rhs.nz;
		nrb 	= rhs.nrb;
		n	= rhs.n;
		ncsb 	= rhs.ncsb;
		blcrange = rhs.blcrange;
		mortoncmp = rhs.mortoncmp;
		diagonal = rhs.diagonal;
		
		nhighbits = rhs.nhighbits;
		nlowbits = rhs.nlowbits;
		highmask = rhs.highmask;
		lowmask = rhs.lowmask;


		if(nz > 0)	// if the copied object is not empty
		{
			num = new NT[nz+2]();  num++;
			bot = new IT[nrb];
			masks = new MTYPE[nrb];
			scansum = new IT[nrb];

			copy ( rhs.num, rhs.num+nz+1, num);
			copy ( rhs.bot, rhs.bot+nrb, bot );
			copy ( rhs.masks, rhs.masks+nrb, masks );
			copy ( rhs.scansum, rhs.scansum+nrb, scansum );
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

template <class NT, class IT, unsigned TTDIM>
BmSym<NT, IT, TTDIM>::~BmSym()
{
	if( nz > 0)
	{
		delete [] scansum;
		delete [] masks;
		delete [] bot;
		delete [] (--num);
	}
	if ( ncsb > 0)
	{
		for(IT i=0; i<ncsb; ++i)
			delete [] top[i];
		delete [] top;
	}
}

template <class NT, class IT, unsigned TTDIM>
BmSym<NT, IT, TTDIM>::BmSym (Csc<NT, IT> & csc, int workers):nz(csc.nz), n(csc.n)
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

                        // i => location of that nonzero in csc.ir and csc.num arrays^M
                        pairarray[k++] = mypair(hindex, ipair(lindex,i));
                }
        }
        sort(pairarray, pairarray+nz);  // sort according to hindex
        SortBlocks(pairarray, csc.num);
        delete [] pairarray;
}

template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val)
{
        typedef pair<IT, pair<IT, IT> > mypair;
        IT cnz = 0;
	IT crb = 0;	// current register block
        IT ldim = IntPower<2>(nhighbits);  // leading dimension (not always equal to ncsb)
	vector<NT> tempnum;
	vector<IT> tempbot;
	vector<MTYPE> M;
        for(IT i = 0; i < ncsb; ++i)
        {
                for(IT j = 0; j < (ncsb-i); ++j)
                {
                        top[i][j] = tempbot.size();	// top points to register blocks
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
			
			int lastregblk = -1;
			for(typename vector<mypair>::iterator itr = blocknz.begin(); itr != blocknz.end(); ++itr) 
                        {
                                tempnum.push_back( val[itr->second.second] );

				int curregblk = getDivident(itr->first, RBSIZE);	
				if(curregblk > lastregblk)	// new register block
				{	
					lastregblk = curregblk;
					M.push_back((MTYPE) 0);
	
					// The following lines implement a get_head function that returns 
					// the top-left index of the register block that this nonzero belongs
    					IT Ci = itr->second.first & lowmask;
    					IT Ri = (itr->second.first >> nlowbits) & lowmask;
					Ci -= getModulo(Ci,RBDIM);
					Ri -= getModulo(Ri,RBDIM);
					IT lefttop = ((lowmask & Ri) << nlowbits) | (lowmask & Ci);	

					tempbot.push_back(lefttop);
				}
				M.back() |= GetMaskTable<MTYPE>(getModulo(itr->first, RBSIZE)); 
                        }
                }
                top[i][ncsb-i] = tempbot.size();
	}

	assert (cnz == nz);
	nz = tempnum.size();	// update the number of off-diagonal nonzeros
	nrb = tempbot.size();	// update the number of off-diagonal register blocks
	masks = new MTYPE[nrb];
	scansum = new IT[nrb];
	bot = new IT[nrb];
	num = new NT[nz+2]();	num++;  // padded for blendv in both sides
	
	copy(M.begin(), M.end(), masks);
	prescan(scansum, masks, nrb);
	copy(tempbot.begin(), tempbot.end(), bot);
	copy(tempnum.begin(), tempnum.end(), num);
}

template<class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::DivideIterationSpace(IT * & lspace, IT * & rspace, IT & lsize, IT & rsize, IT size, IT d) const
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
	else if(d == 2)
	{
		IT chunksfour = size/4;		// we alternate chunks of two
		IT rest = size - 4*chunksfour; 	// rest is modulus 4
		lsize = 2*chunksfour;
		rsize = 2*chunksfour;
		if(rest > 2)	
		{
			rsize += (rest-2);
			lsize += 2;
		}
		else
		{
			lsize += rest;
		}

		lspace = new IT[lsize];
		rspace = new IT[rsize];
	
		for(IT i=0; i<chunksfour; ++i)	// we alternate indices
		{
			lspace[2*i] = 4*i;
			lspace[2*i+1] = 4*i+1;
			rspace[2*i] = 4*i+2;
			rspace[2*i+1] = 4*i+3; 
		}
		if(rest == 3)
		{
			lspace[lsize-2] = size-3;
			lspace[lsize-1] = size-2;
			rspace[rsize-1] = size-1;
		}
		else if(rest == 2)
		{
			lspace[lsize-2] = size-2;		
			lspace[lsize-1] = size-1;
		}
		else if(rest == 1)
		{
			lspace[lsize-1] = size-1;
		}
	}
}

template<class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::MultAddAtomics(NT * __restrict y, const NT * __restrict x, const IT d) const
{
	cilk_for(IT i=0; i< ncsb-d; ++i)	// all blocks at the dth diagonal and beyond
	{
		IT rhi = (i << nlowbits);
		
		cilk_for(IT j=d; j < (ncsb-i); ++j)
		{
			IT chi = ((j+i) << nlowbits);
			symcsr(num+scansum[top[i][j]], masks+top[i][j], bot+top[i][j], top[i][j+1]-top[i][j], x+chi, x+rhi, y+rhi, y+chi, lowmask, nlowbits);
		}
	}
}


template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::MultMainDiag(NT * __restrict y, const NT * __restrict x) const
{
	if(Imbalance(0) > 2 * BALANCETH)	// factor of 2: main diagonal has twice as much parallelism as other diagonals
	{
		cilk_for(IT i=0; i< ncsb; ++i)	// in main diagonal, j = i
		{
			IT hi = (i << nlowbits);
	
			if(i == (ncsb-1) && (n-hi) <= lowmask)	// last iteration and it's irregular (can't parallelize)
			{
				SSEsym(num + scansum[top[i][0]], masks + top[i][0], bot + top[i][0], top[i][1]-top[i][0], x+hi, y+hi, lowmask, nlowbits);
			}
			else
			{
				BlockTriPar(top[i][0], top[i][1], x+hi, y+hi, 0, blcrange, BREAKNRB * (nlowbits+1));
			}
		}
	}
	else	// No need for block parallelization
	{
		cilk_for(IT i=0; i< ncsb; ++i)	// in main diagonal, j = i
		{
			IT hi = (i << nlowbits);
			SSEsym(num + scansum[top[i][0]], masks + top[i][0], bot + top[i][0], top[i][1]-top[i][0], x+hi, y+hi, lowmask, nlowbits);
		}
	}

	const IT diagsize = diagonal.size();
	cilk_for(IT i=0; i < diagsize; ++i)
	{
		y[diagonal[i].first] += diagonal[i].second * x[diagonal[i].first];	// process the diagonal
	}
}


// Multiply the nth block diagonal
// which is composed of blocks A[i][i+n]
template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::MultDiag(NT * __restrict y, const NT * __restrict x, const IT d) const
{
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
		IT start = top[lspace[i]][d];
		IT end = top[lspace[i]][d+1];

		if((top[lspace[i]][d+1] - top[lspace[i]][d] > BALANCETH * lave)	// relative denser block
			&& (!(lspace[i] == (ncsb-d-1) && (n-chi) <= lowmask)))	// and parallelizable
		{
			BlockPar(start, end, x+chi, x+rhi, y+rhi, y+chi, 0, blcrange, BREAKNRB * (nlowbits+1));
		}
		else	
		{
			SSEsym(num + scansum[start], masks + start, bot + start, end-start, x+chi, x+rhi, y+rhi, y+chi, lowmask, nlowbits);
		}
	}
	cilk_for(IT j=0; j< rsize; ++j)
	{
		IT rhi = (rspace[j] << nlowbits) ;
		IT chi = ((rspace[j]+d) << nlowbits);
		IT start = top[rspace[j]][d];
		IT end = top[rspace[j]][d+1];

		if((top[rspace[j]][d+1] - top[rspace[j]][d] > BALANCETH * rave)	// relative denser block
			&& (!(rspace[j] == (ncsb-d-1) && (n-chi) <= lowmask))) // and parallelizable
		{
			BlockPar(start, end, x+chi, x+rhi, y+rhi, y+chi, 0, blcrange, BREAKNRB * (nlowbits+1));
		}
		else
		{
			SSEsym(num + scansum[start], masks + start, bot + start, end-start, x+chi, x+rhi, y+rhi, y+chi, lowmask, nlowbits);
		}
	}
	delete [] lspace;
	delete [] rspace;
}

// Block parallelization for upper triangular compressed sparse blocks
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the `same block
// PRECONDITION: rangeend-rangebeg is a power of two 
template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::BlockTriPar(IT start, IT end, const NT * __restrict subx, NT * __restrict suby, 
				IT rangebeg, IT rangeend, IT cutoff) const
{
	assert(IsPower2(rangeend-rangebeg));
	if(end - start < cutoff)
	{
		SSEsym(num + scansum[start], masks + start, bot + start, end-start, subx, suby, lowmask, nlowbits);
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

		IT ncutoff = std::max<IT>(cutoff/2, MINNRBTOPAR);
	    
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
template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::BlockPar(IT start, IT end, const NT * __restrict subx, const NT * __restrict subx_mirror, 
			NT * __restrict suby, NT * __restrict suby_mirror, IT rangebeg, IT rangeend, IT cutoff) const
{
	assert(IsPower2(rangeend-rangebeg));
	if(end - start < cutoff)
	{
		// Aliasing is not an issue here. BlockPar is only called on off-diagonal register blocks
		SSEsym(num + scansum[start], masks + start, bot + start, end-start, subx, subx_mirror, suby, suby_mirror, lowmask, nlowbits);
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

		IT ncutoff = std::max<IT>(cutoff/2, MINNRBTOPAR);
	    
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
template <class NT, class IT, unsigned TTDIM>
void BmSym<NT, IT, TTDIM>::SeqSpMV(const NT * __restrict x, NT * __restrict y) const
{
	const IT diagsize = diagonal.size();
	for(IT i=0; i < diagsize; ++i)
	{
		y[diagonal[i].first] += diagonal[i].second * x[diagonal[i].first];	// process the diagonal
	}
	for (IT i = 0 ; i < ncsb ; ++i)    // for all block rows of A 
	{
		IT rhi = (i << nlowbits);
		for (IT j = 1 ; j < (ncsb-i) ; ++j)		// for all blocks inside that block row
		{
                	IT chi = ((j+i) << nlowbits);
			SSEsym(num + scansum[top[i][j]], masks+top[i][j], bot+top[i][j], top[i][j+1]-top[i][j], x+chi, x+rhi, y+rhi, y+chi, lowmask, nlowbits);
		}

		SSEsym(num + scansum[top[i][0]], masks+top[i][0], bot+top[i][0], top[i][1]-top[i][0], x+rhi, y+rhi, lowmask, nlowbits);
	}
}

// Imbalance in the dth block diagonal (the main diagonal is the 0th) 
template <class NT, class IT,unsigned TTDIM>
float BmSym<NT, IT,TTDIM>::Imbalance(IT d) const
{
	if(ncsb <= d+1)
	{
		return 0.0;	// no such diagonal exist
	}
        // get the average without the last left-over blockrow
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


// Total number of register blocks in the dth block diagonal (the main diagonal is the 0th) 
template <class NT, class IT, unsigned TTDIM>
IT BmSym<NT, IT,TTDIM>::nrbsum(IT d) const
{
        IT sum = 0; 
	for(size_t i=0; i< ncsb-d; ++i)
        {
		sum += (top[i][d+1] - top[i][d]); 
        }
        return sum;
}

// Print stats to an ofstream object
template <class NT, class IT, unsigned TTDIM>
ofstream & BmSym<NT, IT, TTDIM>::PrintStats(ofstream & outfile) const 
{
	if(nz == 0)
	{
		outfile << "## Matrix Doesn't have any nonzeros" <<endl;
		return outfile;
	}
	const IT ntop = ncsb * ncsb; 	

	outfile << "## Average block is of dimensions "<< lowmask+1 << "-by-" << lowmask+1 << endl;
	outfile << "## Average fill ratio is: " << static_cast<double>(nz) / static_cast<double>((RBSIZE *  nrb)) << endl;
	outfile << "## Number of real blocks is "<< ntop << endl;
	outfile << "## Main (0th) block diagonal imbalance: " << Imbalance(0) << endl;
	outfile << "## 1st block diagonal imbalance: " << Imbalance(1) << endl;
	outfile << "## 2nd block diagonal imbalance: " << Imbalance(2) << endl;

	outfile << "## nrb ratios (block diagonal 0,1,2): " << static_cast<float>(nrbsum(0)) / nrb << ", " 
		<< static_cast<float>(nrbsum(1)) / nrb << ", " << static_cast<float>(nrbsum(2)) / nrb << endl; 
	outfile << "## atomics ratio: " << static_cast<float>(nrb-nrbsum(0)-nrbsum(1)-nrbsum(2))/nrb << endl;
	
	outfile<< "## Total number of nonzeros: " << nz << endl;
	outfile<< "## Total number of register blocks: "<< nrb << endl;
	return outfile;
}


template <class NT, class IT, unsigned TTDIM>
ofstream & BmSym<NT, IT, TTDIM>::Dump(ofstream & outfile) const
{
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
