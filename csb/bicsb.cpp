#include <cassert>
#include "bicsb.h"
#include "utility.h"
#include <cfloat>

// Choose block size as big as possible given the following constraints
// 1) The bot array is addressible by IT
// 2) The parts of x & y vectors that a block touches fits into L2 cache [assuming a saxpy() operation]
// 3) There's enough parallel slackness for block rows (at least SLACKNESS * CILK_NPROC)
template <class NT, class IT>
void BiCsb<NT, IT>::Init(int workers, IT forcelogbeta)
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
      //cout << "Choussing Beta m: " << m << "n: " << n << endl;
      double sqrtn = sqrt(sqrt(static_cast<double>(m) * static_cast<double>(n)));
      IT logbeta = static_cast<IT>(ceil(log2(sqrtn))) + 2;
      if(rowlowbits > logbeta)
	{
	  //cout << "Row Low bits" << endl;
	  rowlowbits = collowbits = logbeta;
	  lowrowmask = lowcolmask = IntPower<2>(logbeta) -1;
	  rowhighbits = rowbits-rowlowbits;
	  colhighbits = colbits-collowbits;
	}
      //cout << "Low row mask:" << lowriwmask << endl;
      // cout << "Beta chosen to be "<< (lowrowmask+1) << endl;
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

// Partial template specialization for booleans
// Does not check cache considerations as this is mostly likely 
// to be used for gaxpy() with multiple rhs vectors (we don't know how many and what type at this point) 
template <class IT>
void BiCsb<bool,IT>::Init(int workers, IT forcelogbeta)
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
      // cout << "Beta chosen to be "<< (lowrowmask+1) << endl;
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


// Constructing empty BiCsb objects (size = 0) are not allowed.
template <class NT, class IT>
BiCsb<NT, IT>::BiCsb (IT size, IT rows, IT cols, int workers): nz(size),m(rows),n(cols)
{
  assert(nz != 0 && n != 0 && m != 0);
  Init(workers);

  num = (NT*) aligned_malloc( nz * sizeof(NT));
  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
}

// Partial template specialization for booleans
template <class IT>
BiCsb<bool, IT>::BiCsb (IT size, IT rows, IT cols, int workers): nz(size),m(rows),n(cols)
{
  assert(nz != 0 && n != 0 && m != 0);
  Init(workers);
  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
}

// copy constructor
template <class NT, class IT>
BiCsb<NT, IT>::BiCsb (const BiCsb<NT,IT> & rhs)
  : nz(rhs.nz), m(rhs.m), n(rhs.n), blcrange(rhs.blcrange), nbr(rhs.nbr), nbc(rhs.nbc), 
    rowhighbits(rhs.rowhighbits), rowlowbits(rhs.rowlowbits), highrowmask(rhs.highrowmask), lowrowmask(rhs.lowrowmask), 
    colhighbits(rhs.colhighbits), collowbits(rhs.collowbits), highcolmask(rhs.highcolmask), lowcolmask(rhs.lowcolmask),
    mortoncmp(rhs.mortoncmp), ispar(rhs.ispar)
{
  if(nz > 0)
    {
      num = (NT*) aligned_malloc( nz * sizeof(NT));
      bot = (IT*) aligned_malloc( nz * sizeof(IT));

      copy (rhs.num, rhs.num + nz, num);	
      copy (rhs.bot, rhs.bot + nz, bot);	
    }
  if ( nbr > 0)
    {
      top = allocate2D<IT>(nbr, nbc+1);
      for(IT i=0; i<nbr; ++i)
	copy (rhs.top[i], rhs.top[i] + nbc + 1, top[i]);
    }
}

// copy constructor for partial NT=boolean specialization
template <class IT>
BiCsb<bool, IT>::BiCsb (const BiCsb<bool,IT> & rhs)
  : nz(rhs.nz), m(rhs.m), n(rhs.n), blcrange(rhs.blcrange), nbr(rhs.nbr), nbc(rhs.nbc), 
    rowhighbits(rhs.rowhighbits), rowlowbits(rhs.rowlowbits), highrowmask(rhs.highrowmask), lowrowmask(rhs.lowrowmask), 
    colhighbits(rhs.colhighbits), collowbits(rhs.collowbits), highcolmask(rhs.highcolmask), lowcolmask(rhs.lowcolmask),
    mortoncmp(rhs.mortoncmp), ispar(rhs.ispar)
{
  if(nz > 0)
    {
      bot = (IT*) aligned_malloc( nz * sizeof(IT));
      copy (rhs.bot, rhs.bot + nz, bot);	
    }
  if ( nbr > 0)
    {
      top = allocate2D<IT>(nbr, nbc+1);
      for(IT i=0; i<nbr; ++i)
	copy (rhs.top[i], rhs.top[i] + nbc + 1, top[i]);
    }
}

template <class NT, class IT>
BiCsb<NT, IT> & BiCsb<NT, IT>::operator= (const BiCsb<NT, IT> & rhs)
{
  if(this != &rhs)		
    {
      if(nz > 0)	// if the existing object is not empty, make it empty
	{
	  aligned_free(bot);
	  aligned_free(num);
	}
      if(nbr > 0)
	{
	  deallocate2D(top, nbr);
	}
      ispar 	= rhs.ispar;
      nz	= rhs.nz;
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
	  num = (NT*) aligned_malloc( nz * sizeof(NT));
	  bot = (IT*) aligned_malloc( nz * sizeof(IT));
	  copy (rhs.num, rhs.num + nz, num);	
	  copy (rhs.bot, rhs.bot + nz, bot);	
	}
      if ( nbr > 0)
	{
	  top = allocate2D<IT>(nbr, nbc+1);
	  for(IT i=0; i<nbr; ++i)
	    copy (rhs.top[i], rhs.top[i] + nbc + 1, top[i]);
	}
    }
  return *this;
}

template <class IT>
BiCsb<bool, IT> & BiCsb<bool, IT>::operator= (const BiCsb<bool, IT> & rhs)
{
  if(this != &rhs)		
    {
      if(nz > 0)	// if the existing object is not empty, make it empty
	{
	  aligned_free(bot);
	}
      if(nbr > 0)
	{
	  deallocate2D(top, nbr);
	}
      ispar 	= rhs.ispar;
      nz	= rhs.nz;
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
	  bot = (IT*) aligned_malloc( nz * sizeof(IT));
	  copy (rhs.bot, rhs.bot + nz, bot);	
	}
      if ( nbr > 0)
	{
	  top = allocate2D<IT>(nbr, nbc+1);
	  for(IT i=0; i<nbr; ++i)
	    copy (rhs.top[i], rhs.top[i] + nbc + 1, top[i]);
	}
    }
  return *this;
}

template <class NT, class IT>
BiCsb<NT, IT>::~BiCsb()
{
  if( nz > 0)
    {
      aligned_free((unsigned char*) num);
      aligned_free((unsigned char*) bot);
    }
  if ( nbr > 0)
    {	
      deallocate2D(top, nbr);	
    }
}

template <class IT>
BiCsb<bool, IT>::~BiCsb()
{
  if( nz > 0)
    {
      aligned_free((unsigned char*) bot);
    }
  if ( nbr > 0)
    {
      deallocate2D(top, nbr);		
    }
}

template <class NT, class IT>
BiCsb<NT, IT>::BiCsb (Csc<NT, IT> & csc, int workers, IT forcelogbeta):nz(csc.nz),m(csc.m),n(csc.n)
{
  typedef std::pair<IT, IT> ipair;
  typedef std::pair<IT, ipair> mypair;
  assert(nz != 0 && n != 0 && m != 0);
  if(forcelogbeta == 0)
    Init(workers);
  else
    Init(workers, forcelogbeta);	

  num = (NT*) aligned_malloc( nz * sizeof(NT));
  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
  mypair * pairarray = new mypair[nz];
  IT k = 0;
  for(IT j = 0; j < n; ++j)
    {
      for (IT i = csc.jc [j] ; i < csc.jc[j+1] ; ++i)	// scan the jth column
	{
	  // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
	  IT hindex = (((highrowmask &  csc.ir[i] ) >> rowlowbits) << colhighbits)
	    | ((highcolmask & j) >> collowbits);
	  IT lindex = ((lowrowmask &  csc.ir[i]) << collowbits) | (lowcolmask & j) ;

	  // i => location of that nonzero in csc.ir and csc.num arrays
	  pairarray[k++] = mypair(hindex, ipair(lindex,i));
	}
    }
  sort(pairarray, pairarray+nz);	// sort according to hindex
  SortBlocks(pairarray, csc.num);
  delete [] pairarray;
}

template <class IT>
template <typename NT>	// to provide conversion from arbitrary Csc<> to specialized BiCsb<bool>
BiCsb<bool, IT>::BiCsb (Csc<NT, IT> & csc, int workers):nz(csc.nz),m(csc.m),n(csc.n)
{
  typedef std::pair<IT, IT> ipair;
  typedef std::pair<IT, ipair> mypair;
  assert(nz != 0 && n != 0 && m != 0);
  Init(workers);
	
  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
  mypair * pairarray = new mypair[nz];
  IT k = 0;
  for(IT j = 0; j < n; ++j)
    {
      for (IT i = csc.jc [j] ; i < csc.jc[j+1] ; ++i)	// scan the jth column
	{
	  // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
	  IT hindex = (((highrowmask &  csc.ir[i] ) >> rowlowbits) << colhighbits)
	    | ((highcolmask & j) >> collowbits);
	  IT lindex = ((lowrowmask &  csc.ir[i]) << collowbits) | (lowcolmask & j) ;

	  // i => location of that nonzero in csc.ir and csc.num arrays
	  pairarray[k++] = mypair(hindex, ipair(lindex,i));
	}
    }
  sort(pairarray, pairarray+nz);	// sort according to hindex
  SortBlocks(pairarray);
  delete [] pairarray;
}

// Assumption: rowindices (ri) and colindices(ci) are "parallel arrays" sorted w.r.t. column index values
template <class NT, class IT>
BiCsb<NT, IT>::BiCsb (IT size, IT rows, IT cols, IT * ri, IT * ci, NT * val, int workers, IT forcelogbeta)
  :nz(size),m(rows),n(cols)
{
  typedef std::pair<IT, IT> ipair;
  typedef std::pair<IT, ipair> mypair;
  assert(nz != 0 && n != 0 && m != 0);
  Init(workers, forcelogbeta);

  num = (NT*) aligned_malloc( nz * sizeof(NT));
  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
  mypair * pairarray = new mypair[nz];
  for(IT k = 0; k < nz; ++k)
    {
      // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
      IT hindex = (((highrowmask &  ri[k] ) >> rowlowbits) << colhighbits)	| ((highcolmask & ci[k]) >> collowbits);	
      IT lindex = ((lowrowmask &  ri[k]) << collowbits) | (lowcolmask & ci[k]) ;

      // k is stored in order to retrieve the location of this nonzero in val array
      pairarray[k] = mypair(hindex, ipair(lindex, k));
    }
  sort(pairarray, pairarray+nz);	// sort according to hindex
  SortBlocks(pairarray, val);
  delete [] pairarray;
}

template <class IT>
BiCsb<bool, IT>::BiCsb (IT size, IT rows, IT cols, IT * ri, IT * ci, int workers, IT forcelogbeta)
  :nz(size),m(rows),n(cols)
{
  typedef std::pair<IT, IT> ipair;
  typedef std::pair<IT, ipair> mypair;
  assert(nz != 0 && n != 0 && m != 0);
  Init(workers, forcelogbeta);

  bot = (IT*) aligned_malloc( nz * sizeof(IT));
  top = allocate2D<IT>(nbr, nbc+1);
  mypair * pairarray = new mypair[nz];
  for(IT k = 0; k < nz; ++k)
    {
      // concatenate the higher/lower order half of both row (first) index and col (second) index bits 
      IT hindex = (((highrowmask &  ri[k] ) >> rowlowbits) << colhighbits)	| ((highcolmask & ci[k]) >> collowbits);	
      IT lindex = ((lowrowmask &  ri[k]) << collowbits) | (lowcolmask & ci[k]) ;

      // k is stored in order to retrieve the location of this nonzero in val array
      pairarray[k] = mypair(hindex, ipair(lindex, k));
    }
  sort(pairarray, pairarray+nz);	// sort according to hindex
  SortBlocks(pairarray);
  delete [] pairarray;
}

template <class NT, class IT>
void BiCsb<NT, IT>::SortBlocks(pair<IT, pair<IT,IT> > * pairarray, NT * val)
{
  typedef typename std::pair<IT, std::pair<IT, IT> > mypair;	
  IT cnz = 0;
  IT ldim = IntPower<2>(colhighbits);	// leading dimension (not always equal to nbc)
  for(IT i = 0; i < nbr; ++i)
    {
      for(IT j = 0; j < nbc; ++j)
	{
	  top[i][j] = cnz;
	  IT prevcnz = cnz; 
	  vector< mypair > blocknz;
	  while(cnz < nz && pairarray[cnz].first == ((i*ldim)+j) )	// as long as we're in this block
	    {
	      IT lowbits = pairarray[cnz].second.first;
	      IT rlowbits = ((lowbits >> collowbits) & lowrowmask);
	      IT clowbits = (lowbits & lowcolmask);
	      IT bikey = BitInterleaveLow(rlowbits, clowbits);
				
	      blocknz.push_back(mypair(bikey, pairarray[cnz++].second));
	    }
	  // sort the block into bitinterleaved order
	  sort(blocknz.begin(), blocknz.end());

	  for(IT k=prevcnz; k<cnz ; ++k)
	    {
	      bot[k] = blocknz[k-prevcnz].second.first;
	      num[k] = val[blocknz[k-prevcnz].second.second];
	    }
	}
      top[i][nbc] = cnz;  // hence equal to top[i+1][0] if i+1 < nbr
    }
  assert(cnz == nz);
}

template <class IT>
void BiCsb<bool, IT>::SortBlocks(pair<IT, pair<IT,IT> > * pairarray)
{
  typedef pair<IT, pair<IT, IT> > mypair;	
  IT cnz = 0;
  IT ldim = IntPower<2>(colhighbits);	// leading dimension (not always equal to nbc)
  for(IT i = 0; i < nbr; ++i)
    {
      for(IT j = 0; j < nbc; ++j)
	{
	  top[i][j] = cnz;
	  IT prevcnz = cnz; 
	  std::vector<mypair> blocknz;
	  while(cnz < nz && pairarray[cnz].first == ((i*ldim)+j) )	// as long as we're in this block
	    {
	      IT lowbits = pairarray[cnz].second.first;
	      IT rlowbits = ((lowbits >> collowbits) & lowrowmask);
	      IT clowbits = (lowbits & lowcolmask);
	      IT bikey = BitInterleaveLow(rlowbits, clowbits);
				
	      blocknz.push_back(mypair(bikey, pairarray[cnz++].second));
	    }
	  // sort the block into bitinterleaved order
	  sort(blocknz.begin(), blocknz.end());

	  for(IT k=prevcnz; k<cnz ; ++k)
	    bot[k] = blocknz[k-prevcnz].second.first;
	}
      top[i][nbc] = cnz;
    }
  assert(cnz == nz);
}

/**
 * @param[IT**] chunks {an array of pointers, ith entry is an address pointing to the top array }
 * 	That address belongs to the the first block in that chunk
 * 	chunks[i] is valid for i = {start,start+1,...,end} 
 *	chunks[0] = btop
 **/ 
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::BMult(IT** chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const
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
	      const RHS * __restrict subx = &x[chi];
	      BlockPar<SR>( *(chunks[start]) , *(chunks[end]), subx, y, 0, blcrange, BREAKEVEN * ysize);
	    }
	  else 		// otherwise block parallelization will fail 
	    {
	      SubSpMV<SR>(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y);
	    }
	}
      else 	// a number of sparse blocks with a total of at most O(\beta) nonzeros
	{
	  SubSpMV<SR>(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y);
	}  
    }
  else
    {
      // divide chunks into half 
      IT mid = (start+end)/2;

      cilk_spawn BMult<SR>(chunks, start, mid, x, y, ysize);
      if(SYNCHED)
	{ 
	  BMult<SR>(chunks, mid, end, x, y, ysize);
	}
      else
	{
	  LHS * temp = new LHS[ysize]();	
	  // not the empty set of parantheses as the initializer, therefore
	  // even if LHS is a built-in type (such as double,int) it will be default-constructed 
	  // The C++ standard says that: A default constructed POD type is zero-initialized,
	  // for non-POD types (such as std::array), the caller should make sure default constructs to zero

	  BMult<SR>(chunks, mid, end, x, temp, ysize);
	  cilk_sync;
			
#pragma simd
	  for(IT i=0; i<ysize; ++i)
	    SR::axpy(temp[i], y[i]);

	  delete [] temp;
	}
    }
}

// partial template specialization for NT=bool
template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::BMult(IT** chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const
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
	      const RHS * __restrict subx = &x[chi];
	      BlockPar<SR>( *(chunks[start]) , *(chunks[end]), subx, y, 0, blcrange, BREAKEVEN * ysize);
	    }
	  else 		// otherwise block parallelization will fail 
	    {
	      SubSpMV<SR>(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y);
	    }
	}
      else 	// a number of sparse blocks with a total of at most O(\beta) nonzeros
	{
	  SubSpMV<SR>(chunks[0], chunks[start]-chunks[0], chunks[end]-chunks[0], x, y);
	}  
    }
  else
    {
      // divide chunks into half 
      IT mid = (start+end)/2;

      cilk_spawn BMult<SR>(chunks, start, mid, x, y, ysize);
      if(SYNCHED)
	{ 
	  BMult<SR>(chunks, mid, end, x, y, ysize);
	}
      else
	{
	  LHS * temp = new LHS[ysize]();	
	  // not the empty set of parantheses as the initializer, therefore
	  // even if LHS is a built-in type (such as double,int) it will be default-constructed 
	  // The C++ standard says that: A default constructed POD type is zero-initialized,
	  // for non-POD types (such as std::array), the caller should make sure default constructs to zero

	  BMult<SR>(chunks, mid, end, x, temp, ysize);
	  cilk_sync;
			
#pragma simd
	  for(IT i=0; i<ysize; ++i)
	    SR::axpy(temp[i], y[i]);					

	  delete [] temp;
	}
    }
}

/**
 * Improved non-zero dividing version of BTransMult (as opposed to spatially dividing)
 * @warning {difference from BMult is that while the top array pointed by chunks is still contiguous... 
 *              the nonzeros pointed by two consecutive top locations - top[i] and top[i+1] are NOT}
 * @param[vector< vector< pair<IT,IT> > * >] chunks {a vector of pointers to vectors of pairs}
 * 	Each vector of pairs is a chunk and each pair is a block within that chunk
 * 	chunks[i] is valid for i = {start,start+1,...,end-1}
 **/
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::BTransMult(vector< vector< tuple<IT,IT,IT> > * > & chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const
{
#ifdef STATS
  blockparcalls += 1;
#endif
  assert(end-start > 0);	// there should be at least one chunk
  if (end-start == 1) 	// single chunk (note that single chunk does not mean single block)
    {
      if(chunks[start]->size() == 1)	// chunk consists of a single (normally dense) block
	{
	  // get the block row id higher order bits to index x (because this is A'x)
	  auto block = chunks[start]->front();    // get the tuple representing this compressed sparse block
	  IT chi = ( get<2>(block) << rowlowbits);
            	
	  // m-chi > lowrowmask for all blocks except the last skinny tall one.
	  // if the last one is regular too, then it has m-chi = lowcolmask+1
	  // parallelize if it is a regular/complete block (and it it is worth it)

	  if(ysize == (lowrowmask+1) && (m-chi) > lowrowmask && (get<1>(block)-get<0>(block)) > BREAKEVEN * ysize)	
	    {
	      const RHS * __restrict subx = &x[chi];
	      BlockParT<SR>( get<0>(block) , get<1>(block), subx, y, 0, blcrange, BREAKEVEN * ysize);
	    }
	  else 		// otherwise block parallelization will fail 
	    {
	      SubSpMVTrans<SR>(*(chunks[start]), x, y);
	    }
	}
      else 	// a number of sparse blocks with a total of at most O(\beta) nonzeros
	{
	  SubSpMVTrans<SR>(*(chunks[start]), x, y);
	}  
    }
  else    // multiple chunks
    {
      IT mid = (start+end)/2;
      cilk_spawn BTransMult<SR>(chunks, start, mid, x, y, ysize);
      if(SYNCHED)
	{
	  BTransMult<SR>(chunks, mid, end, x, y, ysize);
	}
      else
	{
	  LHS * temp = new LHS[ysize]();
	  BTransMult<SR>(chunks, mid, end, x, temp, ysize);
	  cilk_sync;
			
#pragma simd
	  for(IT i=0; i<ysize; ++i)
	    SR::axpy(temp[i], y[i]);					

	  delete [] temp;
	}
    }
}

// Partial template specialization on NT=bool
template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::BTransMult(vector< vector< tuple<IT,IT,IT> > * > & chunks, IT start, IT end, const RHS * __restrict x, LHS * __restrict y, IT ysize) const
{
  assert(end-start > 0);	// there should be at least one chunk
  if (end-start == 1) 	// single chunk (note that single chunk does not mean single block)
    {
      if(chunks[start]->size() == 1)	// chunk consists of a single (normally dense) block
	{
	  // get the block row id higher order bits to index x (because this is A'x)
	  auto block = chunks[start]->front();    // get the tuple representing this compressed sparse block
	  IT chi = ( get<2>(block) << rowlowbits);
            
	  // m-chi > lowrowmask for all blocks except the last skinny tall one.
	  // if the last one is regular too, then it has m-chi = lowcolmask+1
	  if(ysize == (lowrowmask+1) && (m-chi) > lowrowmask )	// parallelize if it is a regular/complete block
	    {
	      const RHS * __restrict subx = &x[chi];
	      BlockParT<SR>( get<0>(block) , get<1>(block), subx, y, 0, blcrange, BREAKEVEN * ysize);
	    }
	  else 		// otherwise block parallelization will fail 
	    {
	      SubSpMVTrans<SR>(*(chunks[start]), x, y);
	    }
	}
      else 	// a number of sparse blocks with a total of at most O(\beta) nonzeros
	{
	  SubSpMVTrans<SR>(*(chunks[start]), x, y);
	}
    }
  else    // multiple chunks
    {
      IT mid = (start+end)/2;
      cilk_spawn BTransMult<SR>(chunks, start, mid, x, y, ysize);
      if(SYNCHED)
	{
	  BTransMult<SR>(chunks, mid, end, x, y, ysize);
	}
      else
	{
	  LHS * temp = new LHS[ysize]();
	  BTransMult<SR>(chunks, mid, end, x, temp, ysize);
	  cilk_sync;
			
#pragma simd
	  for(IT i=0; i<ysize; ++i)
	    SR::axpy(temp[i], y[i]);
            
	  delete [] temp;
	}
    }
}

// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubSpMV(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x[chi];

#ifdef SIMDUNROLL
      IT start = btop[j];
      IT range = (btop[j+1]-btop[j]) >> 2;

      if(range > ROLLING)
	{
	  for (IT k = 0 ; k < range ; ++k)	// for all nonzeros within ith block (expected =~ nnz/n = c)
	    {
	      // ABAB: how to ensure alignment on the stack?
	      // float a[4] __attribute__((aligned(0x1000))); 
#define ALIGN16 __attribute__((aligned(16)))

	      IT ALIGN16 rli4[4]; IT ALIGN16 cli4[4];
	      NT ALIGN16 x4[4]; NT ALIGN16 y4[4];

	      // _mm_srli_epi32: Shifts the 4 signed or unsigned 32-bit integers to right by shifting in zeros.
	      IT pin = start + (k << 2);

	      __m128i bots = _mm_loadu_si128((__m128i*) &r_bot[pin]);	// load 4 consecutive r_bot elements
	      __m128i clis = _mm_and_si128( bots, lcms);
	      __m128i rlis = _mm_and_si128( _mm_srli_epi32(bots, collowbits), lrms);  
	      _mm_store_si128 ((__m128i*) cli4, clis);
	      _mm_store_si128 ((__m128i*) rli4, rlis);

	      x4[0] = subx[cli4[0]];
	      x4[1] = subx[cli4[1]];
	      x4[2] = subx[cli4[2]];
	      x4[3] = subx[cli4[3]];

	      __m128d Y01QW = _mm_mul_pd((__m128d)_mm_loadu_pd(&r_num[pin]), (__m128d)_mm_load_pd(&x4[0]));
	      __m128d Y23QW = _mm_mul_pd((__m128d)_mm_loadu_pd(&r_num[pin+2]), (__m128d)_mm_load_pd(&x4[2]));

	      _mm_store_pd(&y4[0],Y01QW);
	      _mm_store_pd(&y4[2],Y23QW);

	      suby[rli4[0]] += y4[0];
	      suby[rli4[1]] += y4[1];
	      suby[rli4[2]] += y4[2];
	      suby[rli4[3]] += y4[3];
	    }
	  for(IT k=start+4*range; k<btop[j+1]; ++k)
	    {
	      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	      IT cli = (r_bot[k] & lowcolmask);
	      SR::axpy(r_num[k], subx[cli], suby[rli]);
	    }
	}
      else
	{
#endif
	  for(IT k=btop[j]; k<btop[j+1]; ++k)
	    {
	      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	      IT cli = (r_bot[k] & lowcolmask);
	      SR::axpy(r_num[k], subx[cli], suby[rli]);
	    }
#ifdef SIMDUNROLL
	}
#endif
    }
}

// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubSpMV_tar(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x[chi];

#ifdef SIMDUNROLL
      IT start = btop[j];
      IT range = (btop[j+1]-btop[j]) >> 2;

      if(range > ROLLING)
	{
	  for (IT k = 0 ; k < range ; ++k)	// for all nonzeros within ith block (expected =~ nnz/n = c)
	    {
	      // ABAB: how to ensure alignment on the stack?
	      // float a[4] __attribute__((aligned(0x1000))); 
#define ALIGN16 __attribute__((aligned(16)))

	      IT ALIGN16 rli4[4]; IT ALIGN16 cli4[4];
	      NT ALIGN16 x4[4]; NT ALIGN16 y4[4];

	      // _mm_srli_epi32: Shifts the 4 signed or unsigned 32-bit integers to right by shifting in zeros.
	      IT pin = start + (k << 2);

	      __m128i bots = _mm_loadu_si128((__m128i*) &r_bot[pin]);	// load 4 consecutive r_bot elements
	      __m128i clis = _mm_and_si128( bots, lcms);
	      __m128i rlis = _mm_and_si128( _mm_srli_epi32(bots, collowbits), lrms);  
	      _mm_store_si128 ((__m128i*) cli4, clis);
	      _mm_store_si128 ((__m128i*) rli4, rlis);

	      x4[0] = subx[cli4[0]];
	      x4[1] = subx[cli4[1]];
	      x4[2] = subx[cli4[2]];
	      x4[3] = subx[cli4[3]];

	      __m128d Y01QW = _mm_mul_pd((__m128d)_mm_loadu_pd(&r_num[pin]), (__m128d)_mm_load_pd(&x4[0]));
	      __m128d Y23QW = _mm_mul_pd((__m128d)_mm_loadu_pd(&r_num[pin+2]), (__m128d)_mm_load_pd(&x4[2]));

	      _mm_store_pd(&y4[0],Y01QW);
	      _mm_store_pd(&y4[2],Y23QW);

	      suby[rli4[0]] += y4[0];
	      suby[rli4[1]] += y4[1];
	      suby[rli4[2]] += y4[2];
	      suby[rli4[3]] += y4[3];
	    }
	  for(IT k=start+4*range; k<btop[j+1]; ++k)
	    {
	      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	      IT cli = (r_bot[k] & lowcolmask);
	      SR::axpy(r_num[k], subx[cli], suby[rli]);
	    }
	}
      else
	{
#endif
	  for(IT k=btop[j]; k<btop[j+1]; ++k)
	    {
	      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	      IT cli = (r_bot[k] & lowcolmask);
	      SR::axpy(r_num[k], subx[cli], suby[rli]);
	    }
#ifdef SIMDUNROLL
	}
#endif
    }
}

// Partial boolean specialization on NT=bool
template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::SubSpMV(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row or chunk
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x[chi];
      for (IT k = btop[j] ; k < btop[j+1] ; ++k)	// for all nonzeros within ith block (expected =~ nnz/n = c)
	{
	  IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT cli = (r_bot[k] & lowcolmask);
	  SR::axpy(subx[cli], suby[rli]);		// suby [rli] += subx [cli]  where subx and suby are vectors.
	}
    }
}

//! SubSpMVTrans's chunked version
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubSpMVTrans(const vector< tuple<IT,IT,IT> > & chunk, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;
  for(auto itr = chunk.begin(); itr != chunk.end(); ++itr) // over all blocks within this chunk
    {
      // get the starting point for accessing x
      IT chi = ( get<2>(*itr) << rowlowbits);
      const RHS * __restrict subx = &x[chi];
        
      IT nzbeg = get<0>(*itr);
      IT nzend = get<1>(*itr);
        
      for (IT k = nzbeg ; k < nzend ; ++k)
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(r_num[k], subx[cli], suby[rli]);	// suby [rli] += r_num[k] * subx [cli]  where subx and suby are vectors.
	}
    }
}

//! SubSpMVTrans's chunked version with boolean specialization
template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::SubSpMVTrans(const vector< tuple<IT,IT,IT> > & chunk, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  for(auto itr = chunk.begin(); itr != chunk.end(); ++itr)	
    {
      // get the starting point for accessing x
      IT chi = ( get<2>(*itr) << rowlowbits);
      const RHS * __restrict subx = &x[chi];
        
      IT nzbeg = get<0>(*itr);
      IT nzend = get<1>(*itr);
        
      for (IT k = nzbeg ; k < nzend ; ++k)
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(subx[cli], suby[rli]);	// suby [rli] += subx [cli]  where subx and suby are vectors.
	}
    }
}

template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubSpMVTrans(IT col, IT rowstart, IT rowend, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;
  for(IT i= rowstart; i < rowend; ++i)
    {
      // get the starting point for accessing x
      IT chi = (i << rowlowbits);
      const RHS * __restrict subx = &x[chi];
		
      for (IT k = top[i][col] ; k < top[i][col+1] ; ++k)
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(r_num[k], subx[cli], suby[rli]);	// suby [rli] += r_num[k] * subx [cli]  where subx and suby are vectors.
	}
    }	
}


template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::SubSpMVTrans(IT col, IT rowstart, IT rowend, const RHS * __restrict x, LHS * __restrict suby) const
{
  IT * __restrict r_bot = bot;
  for(IT i= rowstart; i < rowend; ++i)
    {
      // get the starting point for accessing x
      IT chi = (i << rowlowbits);
      const RHS * __restrict subx = &x[chi];
      for (IT k = top[i][col] ; k < top[i][col+1] ; ++k)
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(subx[cli], suby[rli]);			// suby [rli] += subx [cli]  where subx and suby are vectors.
	}
    }	
}

// Parallelize the block itself (A*x version)
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the same block
// PRECONDITION: rangeend-rangebeg is a power of two 
// TODO: we rely on the particular implementation of lower_bound for correctness, which is dangerous !
//		 what if lhs (instead of rhs) parameter to the comparison object is the splitter?
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::BlockPar(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
			     IT rangebeg, IT rangeend, IT cutoff) const
{
  assert(IsPower2(rangeend-rangebeg));
  if(end - start < cutoff)
    {
      IT * __restrict r_bot = bot;
      NT * __restrict r_num = num;
      for (IT k = start ; k < end ; ++k)	
	{
	  IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT cli = (r_bot[k] & lowcolmask);
	  SR::axpy(r_num[k], subx[cli], suby[rli]);	// suby [rli] += r_num[k] * subx [cli]  where subx and suby are vectors.
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
	    
      // We can choose to perform [0,3] in parallel and then [1,2] in parallel
      // or perform [0,1] in parallel and then [2,3] in parallel
      // Decision is based on the balance, i.e. we pick the more balanced parallelism
      if( ( absdiff(size0,size3) + absdiff(size1,size2) ) < ( absdiff(size0,size1) + absdiff(size2,size3) ) )
	{	
	  cilk_spawn BlockPar<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockPar<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);			// multiply subblock_3
	  cilk_sync;

	  cilk_spawn BlockPar<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);	// multiply subblock_1
	  BlockPar<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);		// multiply subblock_2
	  cilk_sync;
	}
      else
	{
	  cilk_spawn BlockPar<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockPar<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);	// multiply subblock_1
	  cilk_sync;

	  cilk_spawn BlockPar<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);	// multiply subblock_2
	  BlockPar<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);				// multiply subblock_3
	  cilk_sync;
	}
    }
}


template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::BlockPar(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
			       IT rangebeg, IT rangeend, IT cutoff) const
{
  assert(IsPower2(rangeend-rangebeg));
  if(end - start < cutoff)
    {
      IT * __restrict r_bot = bot;
      for (IT k = start ; k < end ; ++k)	
	{
	  IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT cli = (r_bot[k] & lowcolmask);
	  SR::axpy(subx[cli], suby[rli]);		// suby [rli] += subx [cli]  where subx and suby are vectors.
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
	    
      // We can choose to perform [0,3] in parallel and then [1,2] in parallel
      // or perform [0,1] in parallel and then [2,3] in parallel
      // Decision is based on the balance, i.e. we pick the more balanced parallelism
      if( ( absdiff(size0,size3) + absdiff(size1,size2) ) < ( absdiff(size0,size1) + absdiff(size2,size3) ) )
	{	
	  cilk_spawn BlockPar<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockPar<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);			// multiply subblock_3
	  cilk_sync;

	  cilk_spawn BlockPar<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);	// multiply subblock_1
	  BlockPar<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);		// multiply subblock_2
	  cilk_sync;
	}
      else
	{
	  cilk_spawn BlockPar<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockPar<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);	// multiply subblock_1
	  cilk_sync;

	  cilk_spawn BlockPar<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);	// multiply subblock_2
	  BlockPar<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);				// multiply subblock_3
	  cilk_sync;
	}
    }
}

// Parallelize the block itself (A'*x version)
// start/end: element start/end positions (indices to the bot array)
// bot[start...end] always fall in the same block
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::BlockParT(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
			      IT rangebeg, IT rangeend, IT cutoff) const
{
  if(end - start < cutoff)
    {
      IT * __restrict r_bot = bot;
      NT * __restrict r_num = num;
      for (IT k = start ; k < end ; ++k)	
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(r_num[k], subx[cli], suby[rli]);	// suby [rli] += r_num[k] * subx [cli]  where subx and suby are vectors.
	}
    }
  else
    {
      IT halfrange = (rangebeg+rangeend)/2;
      IT qrt1range = (rangebeg+halfrange)/2;
      IT qrt3range = (halfrange+rangeend)/2;

      // Lower_bound is a version of binary search: it attempts to find the element value in an ordered range [first, last) 
      // Specifically, it returns the first position where value could be inserted without violating the ordering
      IT * mid = std::lower_bound(&bot[start], &bot[end], halfrange, mortoncmp);
      IT * left = std::lower_bound(&bot[start], mid, qrt1range, mortoncmp);
      IT * right = std::lower_bound(mid, &bot[end], qrt3range, mortoncmp);

      /* -------
	 | 0 1 |
	 | 2 3 |
	 ------- */
      // subtracting two pointers pointing to the same array gives you the # of elements separating them
      // we're *sure* that the differences are 1) non-negative, 2) small enough to be indexed by an IT
      IT size0 = static_cast<IT> (left - &bot[start]);
      IT size1 = static_cast<IT> (mid - left);
      IT size2 = static_cast<IT> (right - mid);
      IT size3 = static_cast<IT> (&bot[end] - right);

      IT ncutoff = std::max<IT>(cutoff/2, MINNNZTOPAR);
	    
      // We can choose to perform [0,3] in parallel and then [1,2] in parallel
      // or perform [0,2] in parallel and then [1,3] in parallel
      // Decision is based on the balance, i.e. we pick the more balanced parallelism
      if( ( absdiff(size0,size3) + absdiff(size1,size2) ) < ( absdiff(size0,size2) + absdiff(size1,size3) ) )
	{	
	  cilk_spawn BlockParT<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockParT<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);			// multiply subblock_3
	  cilk_sync;

	  cilk_spawn BlockParT<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);// multiply subblock_1
	  BlockParT<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);		// multiply subblock_2
	  cilk_sync;
	}
      else
	{
	  cilk_spawn BlockParT<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockParT<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);	// multiply subblock_2
	  cilk_sync;

	  cilk_spawn BlockParT<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);// multiply subblock_1
	  BlockParT<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);				// multiply subblock_3
	  cilk_sync;
	}
    }
}


template <class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<bool, IT>::BlockParT(IT start, IT end, const RHS * __restrict subx, LHS * __restrict suby, 
				IT rangebeg, IT rangeend, IT cutoff) const
{
  if(end - start < cutoff)
    {
      IT * __restrict r_bot = bot;
      for (IT k = start ; k < end ; ++k)	
	{
	  // Note the swap in cli/rli
	  IT cli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT rli = (r_bot[k] & lowcolmask);
	  SR::axpy(subx[cli], suby[rli]);		// suby [rli] += subx [cli]  where subx and suby are vectors.
	}
    }
  else
    {
      IT halfrange = (rangebeg+rangeend)/2;
      IT qrt1range = (rangebeg+halfrange)/2;
      IT qrt3range = (halfrange+rangeend)/2;

      // Lower_bound is a version of binary search: it attempts to find the element value in an ordered range [first, last) 
      // Specifically, it returns the first position where value could be inserted without violating the ordering
      IT * mid = std::lower_bound(&bot[start], &bot[end], halfrange, mortoncmp);
      IT * left = std::lower_bound(&bot[start], mid, qrt1range, mortoncmp);
      IT * right = std::lower_bound(mid, &bot[end], qrt3range, mortoncmp);

      /* -------
	 | 0 1 |
	 | 2 3 |
	 ------- */
      // subtracting two pointers pointing to the same array gives you the # of elements separating them
      // we're *sure* that the differences are 1) non-negative, 2) small enough to be indexed by an IT
      IT size0 = static_cast<IT> (left - &bot[start]);
      IT size1 = static_cast<IT> (mid - left);
      IT size2 = static_cast<IT> (right - mid);
      IT size3 = static_cast<IT> (&bot[end] - right);

      IT ncutoff = std::max<IT>(cutoff/2, MINNNZTOPAR);
	    
      // We can choose to perform [0,3] in parallel and then [1,2] in parallel
      // or perform [0,2] in parallel and then [1,3] in parallel
      // Decision is based on the balance, i.e. we pick the more balanced parallelism
      if( ( absdiff(size0,size3) + absdiff(size1,size2) ) < ( absdiff(size0,size2) + absdiff(size1,size3) ) )
	{	
	  cilk_spawn BlockParT<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockParT<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);			// multiply subblock_3
	  cilk_sync;

	  cilk_spawn BlockParT<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);// multiply subblock_1
	  BlockParT<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);		// multiply subblock_2
	  cilk_sync;
	}
      else
	{
	  cilk_spawn BlockParT<SR>(start, start+size0, subx, suby, rangebeg, qrt1range, ncutoff);	// multiply subblock_0
	  BlockParT<SR>(start+size0+size1, end-size3, subx, suby, halfrange, qrt3range, ncutoff);	// multiply subblock_2
	  cilk_sync;

	  cilk_spawn BlockParT<SR>(start+size0, start+size0+size1, subx, suby, qrt1range, halfrange, ncutoff);// multiply subblock_1
	  BlockParT<SR>(end-size3, end, subx, suby, qrt3range, rangeend, ncutoff);				// multiply subblock_3
	  cilk_sync;
	}
    }
}

// Print stats to an ofstream object
template <class NT, class IT>
ofstream & BiCsb<NT, IT>::PrintStats(ofstream & outfile) const 
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
  outfile << "## Col imbalance is " << ColImbalance(*this) << endl;
#ifdef STATS
  outfile << "## Block parallel calls is " << blockparcalls.get_value() << endl;
#endif
  std::vector<int> blocksizes(ntop);
  for(IT i=0; i<nbr; ++i)
    {
      for(IT j=0; j < nbc; ++j) 
	{
	  blocksizes[i*nbc+j] = static_cast<int> (top[i][j+1]-top[i][j]);
	}
    }	
  sort(blocksizes.begin(), blocksizes.end());
  outfile<< "## Total nonzeros: "<< accumulate(blocksizes.begin(), blocksizes.end(), 0) << endl;

  outfile << "## Nonzero distribution (sorted) of blocks follows: \n" ;
  for(IT i=0; i< ntop; ++i)
    {	
      outfile << blocksizes[i] << "\n";
    }
  outfile << endl;
  return outfile;
}

// Print top level statistics to file
template <class NT, class IT>
ofstream & BiCsb<NT, IT>::PrintTopLevel(ofstream & outfile) const 
{
  if(nz == 0)
    {
      outfile << "## Matrix Doesn't have any nonzeros" <<endl;
      return outfile;
    }
  const IT ntop = nbr * nbc; 	
	
  std::vector<int> blocksizes(ntop);
  for(IT i=0; i<nbr; ++i) {
    for(IT j=0; j < nbc; ++j) {
      blocksizes[i*nbc+j] = static_cast<int> (top[i][j+1]-top[i][j]);
    }
  }

  for(IT i=0; i<nbr; ++i) {
    for(IT j=0; j < nbc-1; ++j) {
      outfile << blocksizes[i*nbc+j] << ",";
    }
    outfile << blocksizes[i*nbc+nbc-1] << endl;
  }
  
  return outfile;
}

// Print top level statistics to file in sparse format
template <class NT, class IT>
ofstream & BiCsb<NT, IT>::PrintTopLevelSparse(ofstream & outfile) const 
{
  if(nz == 0)
    {
      outfile << "## Matrix Doesn't have any nonzeros" <<endl;
      return outfile;
    }
  const IT ntop = nbr * nbc; 	
	
  std::vector<int> blocksizes(ntop);
  for(IT i=0; i<nbr; ++i) {
    for(IT j=0; j < nbc; ++j) {
      blocksizes[i*nbc+j] = static_cast<int> (top[i][j+1]-top[i][j]);
    }
  }

  // first row contains top-level size
  outfile << nbr << "," << nbc << "," << endl;
  
  for(IT i=0; i<nbr; ++i) {
    for(IT j=0; j < nbc; ++j) {
      
      if (blocksizes[i*nbc+j] > 0) // if block contains nz elems
        outfile << i << "," << j << "," << blocksizes[i*nbc+j] << endl;
      
    }
  }
  
  return outfile;
}

/////////////////////////////////
// t-SNE kernel Implementation //
// September 2017	       //
// by Kostas Mylonakis	       //
/////////////////////////////////


// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 3;
  
  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[DIM] = {0};
  RHS Yi[DIM] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
  {
    // get higher order bits for column indices
    IT chi = (j << collowbits);
    const RHS * __restrict subx = &x[DIM * chi];

    for(IT k=btop[j]; k<btop[j+1]; ++k)
    {
      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
      IT cli = (r_bot[k] & lowcolmask);

      RHS dist = 0;
      for (int d = 0; d < DIM; ++d) {
        Yi[d] = subx[cli*DIM + d];
        Yj[d] = subxx[rli*DIM + d];
        /* distance computation */
        dist += (Yi[d] - Yj[d])*(Yi[d] - Yj[d]);
      }

      /* P_{iij} \times Q_{ij} */
      const RHS p_times_q = r_num[k] / (1+dist);
      for (int d = 0; d < DIM; ++d)
        suby[rli*DIM + d] += p_times_q * (Yj[d] - Yi[d]);

    }
  }
}

// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel2D(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 2;
  
  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[DIM] = {0};
  RHS Yi[DIM] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
  {
    // get higher order bits for column indices
    IT chi = (j << collowbits);
    const RHS * __restrict subx = &x[DIM * chi];
      
    for(IT k=btop[j]; k<btop[j+1]; ++k)
    {
      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
      IT cli = (r_bot[k] & lowcolmask);

      RHS dist = 0;
      for (int d = 0; d < DIM; ++d) {
        Yi[d] = subx[cli*DIM + d];
        Yj[d] = subxx[rli*DIM + d];
        /* distance computation */
        dist += (Yi[d] - Yj[d])*(Yi[d] - Yj[d]);
      }

      /* P_{iij} \times Q_{ij} */
      const RHS p_times_q = r_num[k] / (1+dist);
      for (int d = 0; d < DIM; ++d)
        suby[rli*DIM + d] += p_times_q * (Yj[d] - Yi[d]);

    }
  }
}

// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel4D(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 4;

  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[DIM] = {0};
  RHS Yi[DIM] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
  {
    // get higher order bits for column indices
    IT chi = (j << collowbits);
    const RHS * __restrict subx = &x[DIM * chi];

    for(IT k=btop[j]; k<btop[j+1]; ++k)
    {
      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
      IT cli = (r_bot[k] & lowcolmask);

      RHS dist = 0;
      for (int d = 0; d < DIM; ++d) {
        Yi[d] = subx[cli*DIM + d];
        Yj[d] = subxx[rli*DIM + d];
        /* distance computation */
        dist += (Yi[d] - Yj[d])*(Yi[d] - Yj[d]);
      }

      /* P_{iij} \times Q_{ij} */
      const RHS p_times_q = r_num[k] / (1+dist);
      for (int d = 0; d < DIM; ++d)
        suby[rli*DIM + d] += p_times_q * (Yj[d] - Yi[d]);

    }
  }
}


// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel1D(IT * __restrict btop, IT bstart, IT bend,
                                    const RHS * __restrict x, LHS * __restrict suby,
                                    IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 1;
  
  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[DIM] = {0};
  RHS Yi[DIM] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
  {
    // get higher order bits for column indices
    IT chi = (j << collowbits);
    const RHS * __restrict subx = &x[DIM * chi];
      
    for(IT k=btop[j]; k<btop[j+1]; ++k)
    {
      IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
      IT cli = (r_bot[k] & lowcolmask);

      Yi[0] = subx[cli];
      Yj[0] = subxx[rli];

      /* distance computation */
      RHS dist = (Yj[0] - Yi[0])*(Yj[0] - Yi[0]);

          
      /* P_{ij} \times Q_{ij} */
      const RHS p_times_q = r_num[k] / (1+dist);
      suby[rli] += p_times_q * (Yj[0] - Yi[0]);

    }
  }
}



template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel_tar(IT * __restrict btop, IT bstart, IT bend, const RHS * __restrict x, LHS * __restrict suby, IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 3;
  
  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[3] = {0};
  RHS Yi[3] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x[DIM * chi];
      
      for(IT k=btop[j]; k<btop[j+1]; ++k)
	{
	  IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT cli = (r_bot[k] & lowcolmask);

    RHS dist = 0;
    cilk_for (int d = 0; d < DIM; ++d) {
      Yi[d] = subx[cli*DIM + d];
      Yj[d] = subxx[rli*DIM + d];
      /* distance computation */
      dist += (Yi[d] - Yj[d])*(Yi[d] - Yj[d]);
    }

	  /* P_{ij} \times Q_{ij} */
	  const RHS p_times_q = r_num[k] / (1+dist);
    for (int d = 0; d < DIM; ++d)
      suby[rli*DIM + d] += p_times_q * (Yj[d] - Yi[d]);


	}
    }
}

template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEkernel(IT * __restrict btop, IT bstart, IT bend,
				  const RHS * __restrict x_row,
				  const RHS * __restrict x_col,
				  LHS * __restrict suby,
				  IT rhi) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  constexpr IT DIM = 3;
  
  const RHS * __restrict subxx = &x_row[DIM*rhi];
  RHS Yj[3] = {0};
  RHS Yi[3] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x_col[DIM * chi];
      
      for(IT k=btop[j]; k<btop[j+1]; ++k)
	{
	  IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
	  IT cli = (r_bot[k] & lowcolmask);

    RHS dist = 0;
	  for(int di=0; di<DIM; di++){
	    Yi[di] = subx[cli*DIM+ di];
	    Yj[di] = subxx[rli*DIM+ di];
      /* distance computation */
	    dist += (Yj[di] - Yi[di])*(Yj[di] - Yi[di]);
	  }

	  /* P_{ij} \times Q_{ij} */
	  const RHS p_times_q = r_num[k] / (1+dist);
	  for (int di = 0 ; di < DIM ; di++)
	    SR::axpy(p_times_q, Yj[di] - Yi[di], suby[rli*DIM+ di]);
	}
    }
}


// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart/bend: block start/end index (to the top array)
template <class NT, class IT>
template <typename SR, typename RHS, typename LHS>
void BiCsb<NT, IT>::SubtSNEcost(IT * __restrict btop, IT bstart, IT bend,
                                const RHS * __restrict x,
                                LHS * __restrict suby,
                                IT rhi,
                                int DIM,
                                double alpha,
                                double zeta) const
{
  IT * __restrict r_bot = bot;
  NT * __restrict r_num = num;

  __m128i lcms = _mm_set1_epi32 (lowcolmask);
  __m128i lrms = _mm_set1_epi32 (lowrowmask);

  const RHS * __restrict subxx = &x[DIM*rhi];
  RHS Yj[10] = {0};
  RHS Yi[10] = {0};
  for (IT j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
    {
      // get higher order bits for column indices
      IT chi = (j << collowbits);
      const RHS * __restrict subx = &x[DIM * chi];
      
      for(IT k=btop[j]; k<btop[j+1]; ++k)
      {
        IT rli = ((r_bot[k] >> collowbits) & lowrowmask);
        IT cli = (r_bot[k] & lowcolmask);

        RHS dist = 0;
        for (int d = 0; d < DIM; ++d) {
          Yi[d] = subx[cli*DIM + d];
          Yj[d] = subxx[rli*DIM + d];
          /*  distance computation */
          dist += (Yi[d] - Yj[d])*(Yi[d] - Yj[d]);
        }

        double p_tmp = alpha * r_num[k];

        const double q_tmp = ( 1.0 / (1.0+dist) ) / zeta;
          
        /* P_{ij} \times Q_{ij} */
        suby[rli] += p_tmp * log( (p_tmp + FLT_MIN) / (q_tmp + FLT_MIN) );
        // for(int d=0; d<DIM; d++){
        // suby[rli*DIM + d] += p_times_q * (Yj[d] - Yi[d]);
        // }


      }
    }
}



/*------------------------------------------------------------
 *
 * AUTHORS
 *
 *   Dimitris Floros                         fcdimitr@auth.gr
 *
 * VERSION
 *
 *   0.2 - December 16, 2017
 *
 * CHANGELOG
 *
 *   0.2 (Dec 16, 2017) - Dimitris
 *      * incorporated TAR and TAR+ codes
 *      
 *   0.1 (Dec 08, 2017) - Dimitris
 *       * added custom function to get top-level statistics
 *
 * ----------------------------------------------------------*/
