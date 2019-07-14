#include "bmcsb.h"
#include "utility.h"

// Choose block size as big as possible given the following constraints
// 1) The bot array is addressible by IT
// 2) The parts of x & y vectors that a block touches fits into L2 cache [assuming a saxpy() operation]
// 3) There's enough parallel slackness for block rows (at least SLACKNESS * CILK_NPROC)
template <class NT, class IT, unsigned TTDIM>
void BmCsb<NT, IT, TTDIM>::Init(int workers, IT forcelogbeta)
{
	ispar = (workers > 1);
	IT roundrowup = nextpoweroftwo(m);
	IT roundcolup = nextpoweroftwo(n);

	// if indices are negative, highestbitset returns -1, 
	// but that will be caught by the sizereq below
	IT rowbits = highestbitset(roundrowup);
	IT colbits = highestbitset(roundcolup);
	bool sizereq;
	if (ispar)
	{
		sizereq = ((IntPower<2>(rowbits) > SLACKNESS * workers) 
			&& (IntPower<2>(colbits) > SLACKNESS * workers));
	}
	else
	{
		sizereq = ((rowbits > 1) && (colbits > 1));
	}
	if(!sizereq)
	{
		cerr << "Matrix too small for this library" << endl;
		return;
	}

	rowlowbits = rowbits-1;	
	collowbits = colbits-1;	
	IT inf = numeric_limits<IT>::max();
	IT maxbits = highestbitset(inf);

	rowhighbits = rowbits-rowlowbits;	// # higher order bits for rows (has at least one bit)
	colhighbits = colbits-collowbits;	// # higher order bits for cols (has at least one bit)
	if(ispar)
	{
		while(IntPower<2>(rowhighbits) < SLACKNESS * workers)
		{
			rowhighbits++;
			rowlowbits--;
		}
	}

	// calculate the space that suby occupies in L2 cache
	IT yL2 = IntPower<2>(rowlowbits) * sizeof(NT);
	while(yL2 > L2SIZE)
	{
		yL2 /= 2;
		rowhighbits++;
		rowlowbits--;
	}

	// calculate the space that subx occupies in L2 cache
	IT xL2 = IntPower<2>(collowbits) * sizeof(NT);
	while(xL2 > L2SIZE)
	{
		xL2 /= 2;
		colhighbits++;
		collowbits--;
	}
	
	// blocks need to be square for correctness (maybe generalize this later?) 
	while(rowlowbits+collowbits > maxbits)
	{
		if(rowlowbits > collowbits)
		{
			rowhighbits++;
			rowlowbits--;
		}
		else
		{
			colhighbits++;
			collowbits--;
		}
	}
	while(rowlowbits > collowbits)
	{
		rowhighbits++;
		rowlowbits--;
	}
	while(rowlowbits < collowbits)
	{
		colhighbits++;
		collowbits--;
	}
	assert (collowbits == rowlowbits);
	lowrowmask = IntPower<2>(rowlowbits) - 1;
	lowcolmask = IntPower<2>(collowbits) - 1;
	if(forcelogbeta != 0)
	{
		IT candlowmask  = IntPower<2>(forcelogbeta) -1;
		cout << "Forcing beta to "<< (candlowmask+1) << " instead of the chosen " << (lowrowmask+1) << endl;
		cout << "Warning : No checks are performed on the beta you have forced, anything can happen !" << endl;
		lowrowmask = lowcolmask = candlowmask;
		rowlowbits = collowbits = forcelogbeta;
		rowhighbits = rowbits-rowlowbits; 
		colhighbits = colbits-collowbits; 
	}
	else 
	{
		double sqrtn = sqrt(sqrt(static_cast<double>(m) * static_cast<double>(n)));
		IT logbeta = static_cast<IT>(ceil(log2(sqrtn))) + 2;
		if(rowlowbits > logbeta)
		{
			rowlowbits = collowbits = logbeta;
			lowrowmask = lowcolmask = IntPower<2>(logbeta) -1;
			rowhighbits = rowbits-rowlowbits;
	                colhighbits = colbits-collowbits;
		}
		cout << "Beta chosen to be "<< (lowrowmask+1) << endl;
	}
	highrowmask = ((roundrowup - 1) ^ lowrowmask);
	highcolmask = ((roundcolup - 1) ^ lowcolmask);
	
	// nbc = #{block columns} = #{blocks in any block row},  nbr = #{block rows)
	IT blcdimrow = lowrowmask + 1;
        IT blcdimcol = lowcolmask + 1;
        nbr = static_cast<IT>(ceil(static_cast<double>(m) / static_cast<double>(blcdimrow)));
        nbc = static_cast<IT>(ceil(static_cast<double>(n) / static_cast<double>(blcdimcol)));
	
	blcrange = (lowrowmask+1) * (lowcolmask+1);	// range indexed by one block
	mortoncmp = MortonCompare<IT>(rowlowbits, collowbits, lowrowmask, lowcolmask);
}


// copy constructor
template <class NT, class IT, unsigned TTDIM>
BmCsb<NT, IT, TTDIM>::BmCsb (const BmCsb<NT,IT,TTDIM> & rhs)
: nz(rhs.nz), m(rhs.m), n(rhs.n), blcrange(rhs.blcrange), nbr(rhs.nbr), nbc(rhs.nbc), nrb(rhs.nrb),
rowhighbits(rhs.rowhighbits), rowlowbits(rhs.rowlowbits), highrowmask(rhs.highrowmask), lowrowmask(rhs.lowrowmask), 
colhighbits(rhs.colhighbits), collowbits(rhs.collowbits), highcolmask(rhs.highcolmask), lowcolmask(rhs.lowcolmask),
mortoncmp(rhs.mortoncmp), ispar(rhs.ispar)
{
	if(nz > 0)  // nz > 0 iff nrb > 0
	{
		num = new NT[nz+2](); num++;
		bot = new IT[nrb];
		masks = new MTYPE[nrb];

		copy ( rhs.num, rhs.num+nz+1, num);
		copy ( rhs.bot, rhs.bot+nrb, bot );
		copy ( rhs.masks, rhs.masks+nrb, masks );
	}
	if ( nbr > 0)
	{
		top = new IT* [nbr];
		for(IT i=0; i<nbr; ++i)
			top[i] = new IT[nbc+1]; 
		for(IT i=0; i<nbr; ++i)
			for(IT j=0; j <= nbc; ++j) 
				top[i][j] = rhs.top[i][j];
	}
}

template <class NT, class IT, unsigned TTDIM>
BmCsb<NT, IT, TTDIM> & BmCsb<NT, IT, TTDIM>::operator= (const BmCsb<NT, IT, TTDIM> & rhs)
{
	if(this != &rhs)		
	{
		if(nz > 0)	// if the existing object is not empty
		{
			// make it empty
			delete [] masks;
			delete [] bot;
			delete [] (--num);
		}
		if(nbr > 0)
		{
			for(IT i=0; i<nbr; ++i)
				delete [] top[i];
			delete [] top;
		}

		ispar 	= rhs.ispar;
		nz	= rhs.nz;
		nrb  	= rhs.nrb;
		n	= rhs.n;
		m   	= rhs.m;
		nbr 	= rhs.nbr;
		nbc 	= rhs.nbc;
		blcrange = rhs.blcrange;

		rowhighbits = rhs.rowhighbits;
		rowlowbits = rhs.rowlowbits;
		highrowmask = rhs.highrowmask;
		lowrowmask = rhs.lowrowmask;

		colhighbits = rhs.colhighbits;
		collowbits = rhs.collowbits;
		highcolmask = rhs.highcolmask;
		lowcolmask= rhs.lowcolmask;
		mortoncmp = rhs.mortoncmp;

		if(nz > 0)	// if the copied object is not empty
		{
			num = new NT[nz+2]();  num++;
			bot = new IT[nrb];
			masks = new MTYPE[nrb];

			copy ( rhs.num, rhs.num+nz+1, num);
			copy ( rhs.bot, rhs.bot+nrb, bot );
			copy ( rhs.masks, rhs.masks+nrb, masks );
		}
		if(nbr > 0)
		{
			top = new IT* [nbr];
			for(IT i=0; i<nbr; ++i)
				top[i] = new IT[nbc+1]; 
			for(IT i=0; i<nbr; ++i)
				for(IT j=0; j <= nbc; ++j) 
					top[i][j] = rhs.top[i][j];
		}
	}
	return *this;
}

template <class NT, class IT, unsigned TTDIM>
BmCsb<NT, IT, TTDIM>::~BmCsb()
{
	if( nz > 0)
	{
		delete [] masks;
		delete [] bot;
		delete [] (--num);
	}
	if ( nbr > 0)
	{
		for(IT i=0; i<nbr; ++i)
			delete [] top[i];
		delete [] top;
	}
}

template <class NT, class IT, unsigned TTDIM>
BmCsb<NT, IT, TTDIM>::BmCsb (Csc<NT, IT> & csc, int workers):nz(csc.nz), m(csc.m),n(csc.n)
{
        typedef std::pair<IT, IT> ipair;
        typedef std::pair<IT, ipair> mypair;

        assert(nz != 0 && n != 0 && m != 0);
        Init(workers);

        num = new NT[nz+2]();	num++;  // Padding for SSEspmv (the blendv operation)
	// bot is later to be resized to nrb (number of register blocks)
	// nrb < nz as the worst case happens when each register block contains only one nonzero

        top = allocate2D<IT>(nbr, nbc+1);
        mypair * pairarray = new mypair[nz];
        IT k = 0;
        for(IT j = 0; j < n; ++j)
        {
                for (IT i = csc.jc [j] ; i < csc.jc[j+1] ; ++i) // scan the jth column
                {
                        // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
                        IT hindex = (((highrowmask &  csc.ir[i] ) >> rowlowbits) << colhighbits)
                                                                                | ((highcolmask & j) >> collowbits);
                        IT lindex = ((lowrowmask &  csc.ir[i]) << collowbits) | (lowcolmask & j) ;

                        // i => location of that nonzero in csc.ir and csc.num arrays^M
                        pairarray[k++] = mypair(hindex, ipair(lindex,i));
                }
        }
        sort(pairarray, pairarray+nz);  // sort according to hindex
        SortBlocks(pairarray, csc.num);
        delete [] pairarray;
}

template <class NT, class IT, unsigned TTDIM>
void BmCsb<NT, IT, TTDIM>::SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val)
{
        typedef pair<IT, pair<IT, IT> > mypair;
        IT cnz = 0;
	IT crb = 0;	// current register block
        IT ldim = IntPower<2>(colhighbits);  // leading dimension (not always equal to nbc)
	vector<IT> tempbot;
	vector<MTYPE> M;
        for(IT i = 0; i < nbr; ++i)
        {
                for(IT j = 0; j < nbc; ++j)
                {
                        top[i][j] = tempbot.size();	// top array now points to register blocks (instead of nonzeros)
                        IT prevcnz = cnz;
                        std::vector<mypair> blocknz;
                        while(cnz < nz && pairarray[cnz].first == ((i*ldim)+j) )        // as long as we're in this block
                        {
                                IT lowbits = pairarray[cnz].second.first;
                                IT rlowbits = ((lowbits >> collowbits) & lowrowmask);
                                IT clowbits = (lowbits & lowcolmask);
                                IT bikey = BitInterleaveLow(rlowbits, clowbits);

                                blocknz.push_back(mypair(bikey, pairarray[cnz++].second));
                        }
                        // sort the block into bitinterleaved order
                        sort(blocknz.begin(), blocknz.end());

			int lastregblk = -1;
			IT bnz = blocknz.size();

			for(IT bcur=0; bcur < bnz; ++bcur)
			{
				int curregblk = getDivident(blocknz[bcur].first, RBSIZE);	
				if(curregblk > lastregblk)	// new register block
				{	
					lastregblk = curregblk;
					M.push_back((MTYPE) 0);
	
					// The following lines implement a get_head function that returns 
					// the top-left index of the register block that this nonzero belongs
    					IT Ci = blocknz[bcur].second.first & lowcolmask;
    					IT Ri = (blocknz[bcur].second.first >> collowbits) & lowrowmask;
					Ci -= getModulo(Ci,RBDIM);
					Ri -= getModulo(Ri,RBDIM);
					IT lefttop = ((lowrowmask & Ri) << collowbits) | (lowcolmask & Ci);	

					tempbot.push_back(lefttop);
				}
				M.back() |= GetMaskTable<MTYPE>(getModulo(blocknz[bcur].first, RBSIZE)); 
			}
                        for(IT k=prevcnz; k<cnz ; ++k)
                        {
                                num[k] = val[blocknz[k-prevcnz].second.second];
                        }
                }
                top[i][nbc] = tempbot.size();
        }
	assert(M.size() == tempbot.size());
	masks = new MTYPE[M.size()];
	copy(M.begin(), M.end(), masks);
	
	bot = new IT[tempbot.size()];
	copy(tempbot.begin(), tempbot.end(), bot);
	nrb = tempbot.size(); 	

        assert(cnz == nz);
}



/**
  * @param[IT**] chunks {an array of pointers, ith entry is an address pointing to the top array }
  * 	That address belongs to the the first block in that chunk
  * 	chunks[i] is valid for i = {start,start+1,...,end} 
  *	chunks[0] = btop
  **/ 
template <class NT, class IT, unsigned TTDIM>
void BmCsb<NT, IT, TTDIM>::BMult(IT** chunks, IT start, IT end, const NT * x, NT * y, IT ysize, IT * __restrict sumscan) const
{
	assert(end-start > 0);	// there should be at least one chunk
	if (end-start == 1) 	// single chunk
	{
		if((chunks[end] - chunks[start]) == 1)	// chunk consists of a single (normally dense) block 
		{
			IT chi = ( (chunks[start] - chunks[0])  << collowbits);

			// m-chi > lowcolmask for all blocks except the last skinny tall one.
			// if the last one is regular too, then it has m-chi = lowcolmask+1
			if(ysize == (lowrowmask+1) && (m-chi) > lowcolmask )	// parallelize if it is a regular/complete block 	
			{
				const NT * __restrict subx = &x[chi];
				BlockPar( *(chunks[start]) , *(chunks[end]), subx, y, 0, blcrange, BREAKNRB * ysize, sumscan);
			}
			else 		// otherwise block parallelization will fail 
			{
				SubSpMV(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y, sumscan);
			}
		}
		else 	// a number of sparse blocks with a total of at most O(\beta) nonzeros
		{
			SubSpMV(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y, sumscan);
		}  
	}
	else
	{
		IT mid = (start+end)/2;                 // divide chunks into half 
		cilk_spawn BMult(chunks, start, mid, x, y, ysize, sumscan);
		if(SYNCHED)
		{ 
			BMult(chunks, mid, end, x, y, ysize, sumscan);
		}
		else
		{
			NT * temp = new NT[ysize];
			std::fill_n(temp, ysize, 0.0);
			BMult(chunks, mid, end, x, temp, ysize, sumscan);
			cilk_sync;
			for(IT i=0; i<ysize; ++i)
				y[i] += temp[i];
			delete [] temp;
		}
	}
}



// Parallelize the block itself (A*x version)
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the same block
// PRECONDITION: rangeend-rangebeg is a power of two 
// TODO: we rely on the particular implementation of lower_bound for correctness, which is dangerous !
//		 what if lhs (instead of rhs) parameter to the comparison object is the splitter?
template <class NT, class IT, unsigned TTDIM>
void BmCsb<NT, IT, TTDIM>::BlockPar(IT start, IT end, const NT * __restrict subx, NT * __restrict suby, 
				IT rangebeg, IT rangeend, IT cutoff, IT * __restrict sumscan) const
{
	assert(IsPower2(rangeend-rangebeg));
	if(end - start < cutoff)
	{
		SSEspmv(num + sumscan[start], masks + start, bot + start, end-start, subx, suby, lowcolmask, lowrowmask, collowbits);
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
	    
		// We can choose to perform [0,3] in parallel and then [1,2] in parallel
		// or perform [0,1] in parallel and then [2,3] in parallel
		// Decision is based on the balance, i.e. we pick the more balanced parallelism
		if( ( absdiff(size0,size3) + absdiff(size1,size2) ) < ( absdiff(size0,size1) + absdiff(size2,size3) ) )
		{	
			cilk_spawn BlockPar(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff,sumscan);	// multiply subblock_0
			BlockPar(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff,sumscan);			// multiply subblock_3
			cilk_sync;

			cilk_spawn BlockPar(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff,sumscan);	// multiply subblock_1
			BlockPar(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff,sumscan);		// multiply subblock_2
			cilk_sync;
		}
		else
		{
			cilk_spawn BlockPar(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff,sumscan);	// multiply subblock_0
			BlockPar(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff,sumscan);	// multiply subblock_1
			cilk_sync;

			cilk_spawn BlockPar(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff,sumscan);	// multiply subblock_2
			BlockPar(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff,sumscan);				// multiply subblock_3
			cilk_sync;
		}
	}
}


// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT, unsigned TTDIM>
void BmCsb<NT, IT, TTDIM>::SubSpMV(IT * __restrict btop, IT bstart, IT bend, const NT * __restrict x, NT * __restrict suby, IT * __restrict sumscan) const
{
	for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
	{
		IT chi = (j << collowbits);  // &x[chi] addresses the higher order bits for column indices

		if(btop[j+1] - btop[j] > 0)
		{
			SSEspmv(num + sumscan[btop[j]], masks + btop[j], bot + btop[j], btop[j+1]-btop[j], x+chi, suby, lowcolmask, lowrowmask, collowbits);
		}
	}
}


// Print stats to an ofstream object
template <class NT, class IT, unsigned TTDIM>
ofstream & BmCsb<NT, IT, TTDIM>::PrintStats(ofstream & outfile) const 
{
	if(nz == 0)
	{
		outfile << "## Matrix Doesn't have any nonzeros" <<endl;
		return outfile;
	}
	const IT ntop = nbr * nbc; 	

	outfile << "## Average block is of dimensions "<< lowrowmask+1 << "-by-" << lowcolmask+1 << endl;
	outfile << "## Number of real blocks is "<< ntop << endl;
	outfile << "## Row imbalance is " << RowImbalance(*this) << endl;
	
	std::vector<int> blocksizes(ntop);
	for(IT i=0; i<nbr; ++i)
	{
		for(IT j=0; j < nbc; ++j) 
		{
			blocksizes[i*nbc+j] = static_cast<int> (top[i][j+1]-top[i][j]);
		}
	}	
	sort(blocksizes.begin(), blocksizes.end());
	outfile<< "## Total number of nonzeros: " << nz << endl;
	outfile<< "## Total number of register blocks: "<< accumulate(blocksizes.begin(), blocksizes.end(), 0) << endl;
	outfile<< "## Average fill ratio is: " << static_cast<double>(nz) / static_cast<double>((RBSIZE *  nrb)) << endl;
	outfile<< "## The histogram of fill ratios within register blocks:" << endl;
	
	unsigned * counts = new unsigned[nrb];
	popcountall(masks, counts, nrb); 
	printhistogram(counts, nrb, RBSIZE);
	delete [] counts;

	outfile << "## Nonzero distribution (sorted) of blocks follows: \n" ;
	for(IT i=0; i< ntop; ++i)
	{	
		outfile << blocksizes[i] << "\n";
	}
	outfile << endl;
	return outfile;
}
