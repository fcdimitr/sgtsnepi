/**
 * @file   bssb.cpp
 * @author Nikos Sismanis <nsismani@auth.gr>
 * @date   Wed Sep 19 13:51:08 2017
 * 
 * @brief  BSSB core functions
 *
 * Core function for BSSB implementation. Not all of them are visible
 * through the corresponding header
 * 
 * 
 */

#include "benchmark_csb.hpp"

#include "orders.hpp"

#include "bssb.hpp"


bool operator<(const boxHelper &lhs, const boxHelper &rhs) {
  return lhs.size < rhs.size;
}


/* PQ Kernel */
void computeLeaf(sparseBlock* Ps, double* F, double* Y, int dim){

  double         *vv = Ps->vv;
  int32_t        *ju = Ps->ju;
  int32_t        *ii = Ps->ii;
  int32_t        *li = Ps->li;
  int32_t         m  = Ps->Nrow;
  int32_t         n  = Ps->Ncol;

  int32_t nn = Ps->nuj;

  int32_t ss = 0;

  int32_t R = Ps->row;
  int32_t C = Ps->col;

	       
  double* Y0i = &Y[R * dim];
  double* Y0j = &Y[C * dim];

  double* F0i = &F[R * dim];



  for (uint32_t j = 0; j < nn; j++) {

    double accum[DIM] = {0};
    double Ftemp[DIM] = {0};
    double Yj[DIM] = {0};
    double Yi[DIM] = {0};

    const int32_t k = li[j];    /* number of nonzero elements of each column */

    
    Yj[:] = Y0j[ ju[j]*dim + 0:dim ];

    accum[:] = 0;

    /* for each non zero element */
    for (uint32_t idx = 0; idx < k; idx++) {

      const uint32_t i = (ii[ss + idx]);

      Yi[:] = Y0i[ i * dim + 0:dim ];

      /* distance computation */
      double dist = __sec_reduce_add( (Yj[:] - Yi[:])*(Yj[:] - Yi[:]) );

      // FILE *f_i = fopen( "csc_i.bin", "ab" );
      // FILE *f_j = fopen( "csc_j.bin", "ab" );
      // FILE *f_v = fopen( "csc_v.bin", "ab" );

      // int    i_bin = i+R;
      // int    j_bin = ju[j]+C;
      // double v_bin = vv[ss+idx];
      
      // fwrite( &i_bin, sizeof(i_bin), 1, f_i );
      // fwrite( &j_bin, sizeof(j_bin), 1, f_j );
      // fwrite( &v_bin, sizeof(v_bin), 1, f_v );

      // fclose( f_i ); fclose( f_j ); fclose( f_v );
      
      /* P_{ij} \times Q_{ij} */
      double p_times_q = vv[ss+idx] / (1+dist);

      Ftemp[:] = p_times_q * ( Yj[:] - Yi[:] );

      /* F_{attr}(i,j) */
      F0i[i*dim + 0:dim] -= Ftemp[:];
    }

    ss += k;

  }
    
    
}


/* SpMV Multiplication Kernel */
void blockMatMult(sparseBlock* Ps, double* F, double* Y){

  double         *vv = Ps->vv;
  int32_t        *ju = Ps->ju;
  int32_t        *ii = Ps->ii;
  int32_t        *li = Ps->li;
  int32_t         m = Ps->Nrow;
  int32_t         n = Ps->Ncol;

  int32_t nn = Ps->nuj;

  int32_t ss = 0;

  int32_t R = Ps->row;
  int32_t C = Ps->col;


  double* Y0i = &Y[R];
  double* Y0j = &Y[C];

  double* F0i = &F[R];


  for (uint32_t j = 0; j < nn; j++) {

    double accum = 0;
    double Ftemp = 0;
    double Yj = 0;
    double Yi = 0;

    Yj = Y0j[ ju[j] ];

    
    const int32_t k = li[j];
    for (uint32_t idx = 0; idx < k; idx++) {
      
      const uint32_t i = (ii[ss + idx]);

      double pval = vv[ ss + idx];

      Ftemp = pval * Yj;	

      F0i[i] += Ftemp; 
    }
    ss += k;
  }


}

/**
 * Compute B = A for CSR matrix A, CSC matrix B
 *
 * Also, with the appropriate arguments can also be used to:
 *   - compute B = A^t for CSR matrix A, CSR matrix B
 *   - compute B = A^t for CSC matrix A, CSC matrix B
 *   - convert CSC->CSR
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *
 * Output Arguments:
 *   I  Bp[n_col+1] - column pointer
 *   I  Bj[nnz(A)]  - row indices
 *   T  Bx[nnz(A)]  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, Bx must be preallocated
 *
 * Note: 
 *   Input:  column indices *are not* assumed to be in sorted order
 *   Output: row indices *will be* in sorted order
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 * 
 */
template <class I, class T>
void csr_tocsc(const I n_row,
	       const I n_col, 
	       const I Ap[], 
	       const I Aj[], 
               const T Ax[],
                     I Bp[],
                     I Bi[],
                     T Bx[])
{  
    const I nnz = Ap[n_row];

    //compute number of non-zero entries per column of A 
    std::fill(Bp, Bp + n_col, 0);

    for (I n = 0; n < nnz; n++){            
        Bp[Aj[n]]++;
    }

    //cumsum the nnz per column to get Bp[]
    for(I col = 0, cumsum = 0; col < n_col; col++){     
        I temp  = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz; 

    for(I row = 0; row < n_row; row++){
        for(I jj = Ap[row]; jj < Ap[row+1]; jj++){
            I col  = Aj[jj];
            I dest = Bp[col];

            Bi[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col]++;
        }
    }  

    for(I col = 0, last = 0; col <= n_col; col++){
        I temp  = Bp[col];
        Bp[col] = last;
        last    = temp;
    }
}   

void csc2csr_top(top_lvl_csc *BSSB_CSC, int nCol, int nRow){

  int *csc_jc = BSSB_CSC->jc;
  int *csc_ir = BSSB_CSC->ir;
  sparseBlock *csc_sb = BSSB_CSC->Pb;
  
  int nnz = csc_jc[nCol];
  
  int *csr_jr = (int *) malloc((nCol+1)*sizeof(int));
  int *csr_ic = (int *) malloc(nnz*sizeof(int));
  sparseBlock *csr_sb = (sparseBlock *) malloc(nnz*sizeof(sparseBlock));

  csr_tocsc<int, sparseBlock>(nRow,
                              nCol, 
                              csc_jc, 
                              csc_ir, 
                              csc_sb,
                              csr_jr,
                              csr_ic,
                              csr_sb);

  BSSB_CSC->jc = csr_jr;
  BSSB_CSC->ir = csr_ic;
  BSSB_CSC->Pb = csr_sb;

  free( csc_jc );
  free( csc_ir );
  free( csc_sb );

}

void updateLeafOrder(node_t *box,
		     int32_t *leafMap,
		     int32_t *leafStart,
		     int32_t *leafSize){

  if(box->numChild == 0){
    box->leafId = leafMap[box->strParticle];
    //leafStart[box->leafId] = box->strParticle;
    //leafSize[box->leafId] = box->pop;
  } else{

    node_t *cs = box->first_child;
    for(int i=0; i<box->numChild; i++){
      updateLeafOrder(cs, leafMap, leafStart, leafSize);
      cs = cs->next_sibling;
    }
    
  }
}

void updateLeafOrder(int32_t *leafMap,
		     int32_t *leafStart,
		     int32_t *leafSize,
		     int N){

  int offset = 0;
  int leafCount = 0;
  for(int i=0; i<N; i+=offset){
    offset = leafMap[i];
    leafStart[leafCount] = i;
    leafSize[leafCount] = offset;
    leafCount++;
  }

}

void createLeafMap(int32_t *leafMap, node_t *box){

  if(box->numChild == 0){

    leafMap[box->strParticle] = box->pop;
    
  } else {


    node_t *cs = box->first_child;	
    for(int i=0; i<box->numChild; i++){
      createLeafMap(leafMap, cs);
      cs = cs->next_sibling;	
    }
    
  }
  

}


void fillMap(int32_t *col2leaves, int N, int f){

  for(int i=0; i<N; i++){
    col2leaves[i] = f;
  }

}


int32_t mapClo2Leaves(int32_t *col2leaves,
		      int32_t *leafMap,
		      int N){

  int offset = 0;
  int pop = 0;
  int leafCount = 0;
  for(int i=0; i<N; i += offset){

    pop += leafMap[i];
    offset = leafMap[i];
    
    fillMap(&col2leaves[i], leafMap[i], leafCount);
    leafCount++;	
  }	
  
  return leafCount;
}


/* Count the number of nnz in every block of the sparse top level, 
   assuming CSC packing of the interaction matrix */
void count_block_nnz_sptop(top_lvl_csc *Ps,
			   int32_t *rowMap,
			   int32_t *colMap,
			   int32_t *colStart,
			   int32_t *colSizes,
			   int *ir,
			   int *jc,
			   int Ncols,
			   int nRowBlocks,
			   int nnz){


  Ps->ir = (int32_t *)malloc(nnz * sizeof(int32_t));
  Ps->Pb = (sparseBlock *)malloc(nnz * sizeof(sparseBlock));

  /* initialize with zeros */
  for(int i=0; i<nnz; i++) {Ps->Pb[i].nnz = 0;}

  /* Thraverse every column block */
  for(int bi=0; bi<nRowBlocks; bi++){


    int cursor = 0;
    int32_t *rowCount = (int32_t *)calloc(nRowBlocks , sizeof(int32_t)); // Create a bense block column 
    
    int cStr = colStart[bi]; // Start of the column block
    int cStp = cStr + colSizes[bi]; // End of the column block
    
    int blkstr = Ps->jc[bi];  // Start of the compressed block column in csc packing 

    for(int j=cStr; j<cStp; j++){

      const int k = jc[j+1] - jc[j];

      for (unsigned int idx = 0; idx < k; idx++) {

	const unsigned int i = ir[ jc[j] + idx ] ;
	int rowBlockId = rowMap[i];
	rowCount[rowBlockId]++;

      }

    }

    /* Scan the boxes */
    /* Sparsify the dense block column */
    for(int i=0; i<nRowBlocks; i++){

      if (rowCount[i] > 0){ // if the block is non-empty
	Ps->Pb[blkstr+cursor].nnz = rowCount[i];
	Ps->ir[blkstr+cursor] = i;
	cursor++;
      }
      
    }

    free(rowCount);
  }
  
}


/* Finish the construction of every block and 
   turn it to CSC2 packing*/
void finishBlock(sparseBlock *Ps){

  /* Find unique columns */
  int32_t nCols = Ps->Ncol;
  int32_t nRows = Ps->Nrow;

  int32_t *lj = (int32_t *)calloc(nCols, sizeof(int32_t));
  int32_t *li = (int32_t *)calloc(nRows, sizeof(int32_t));

  /* Count volumns that have nnz elements */
  for(int i=0; i<Ps->nnz; i++){
    lj[Ps->jj[i]]++;
    li[Ps->ii[i]]++;
  }


  /* Count the unique */  
  int nuj = 0;
  for(int i=0; i<nCols; i++){
    if(lj[i] > 0){
      nuj++;
    }
  }


  int nui = 0;
  for(int i=0; i<nRows; i++){
    if(li[i] > 0){
      nui++;
    }
  }
  
  /* Allocate space for the unique */
  Ps->nuj = nuj; Ps->nui = nui;
  Ps->ju = (int32_t *)malloc(nuj * sizeof(int32_t)); // unique columns
  Ps->iu = (int32_t *)malloc(nui * sizeof(int32_t)); // unique rows 
  Ps->li = (int32_t *)malloc(nuj * sizeof(int32_t)); // Histogram of columns	

  /* Move unique elements */
  nuj = 0; nui = 0;
  /* gather unique columns */ 
  for(int j=0; j<nCols; j++){
    if(lj[j] > 0){
      Ps->ju[nuj] = j;
      Ps->li[nuj] = lj[j];
      nuj++;
    }
  }

  /* gather unique rows */
  for(int i=0; i<nRows; i++){
    if(li[i] > 0){
      Ps->iu[nui] = i;
      nui++;
    }
  }
  
  free(lj);
  free(li);

}

/* Sparse top level */
void formSparseBlockSparseTop(top_lvl_csc *Ps,
			      int32_t *rowMap,
			      int32_t *colMap,
			      int32_t *rowStart,
			      int32_t *colStart,
			      int32_t *rowSize,
			      int32_t *colSize,
			      int *ir,
			      int *jc,
			      double *vv,
			      int Ncols,
			      int nRowBlocks,
			      int nColBlocks,
			      int nnz){


  /* Block memory allocation */
  for(int j=0; j<nColBlocks; j++){

    const int k = Ps->jc[j+1] - Ps->jc[j];

    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = Ps->jc[j] + idx;
      const unsigned int i = Ps->ir[blk];
     

      int bnnz = Ps->Pb[blk].nnz;
      Ps->Pb[blk].ii = (int32_t *)malloc(bnnz * sizeof(int32_t));
      Ps->Pb[blk].vv = (double *)malloc(bnnz * sizeof(double));
      Ps->Pb[blk].jj = (int32_t *)malloc(bnnz * sizeof(int32_t));
      
      Ps->Pb[blk].row = rowStart[i];
      Ps->Pb[blk].col = colStart[j];

      Ps->Pb[blk].Nrow = rowSize[i];
      Ps->Pb[blk].Ncol = colSize[j];
    }

  }


  /* scan and assign the points */
  for(int bi=0; bi<nColBlocks; bi++){

    int cursor = 0;
    int32_t *rowCount = (int32_t *)calloc(nRowBlocks , sizeof(int32_t));
    int32_t *ptr = (int32_t *)calloc(nRowBlocks, sizeof(int32_t));
    sparseBlock *rowBlocks = (sparseBlock *)malloc(nRowBlocks * sizeof(sparseBlock));

    //for(int i=0; i<nRowBlocks; i++) rowBlocks[i].nnz = 0;
    
    int cStr = colStart[bi];
    int cStp = cStr + colSize[bi];

    int blkstr = Ps->jc[bi];

    /* fisrt scan to count */
    for(int j=cStr; j<cStp; j++){
      const int k = jc[j+1] - jc[j];
      

      for (unsigned int idx = 0; idx < k; idx++) {

	const unsigned int i = (ir[jc[j] + idx]);
	int rowBlockId = rowMap[i];
	rowCount[rowBlockId]++;
      }

    }

    /* second scan to assing */
    for(int i=0; i<nRowBlocks; i++){

      if (rowCount[i] > 0){
	/*
	rowBlocks[cursor].ii = (int32_t *)malloc(rowCount[i]*sizeof(int32_t));
	rowBlocks[cursor].jj = (int32_t *)malloc(rowCount[i]*sizeof(int32_t));
	rowBlocks[cursor].vv = (double *)malloc(rowCount[i]*sizeof(double));
	rowBlocks[cursor].nnz = rowCount[i];
	*/
	rowBlocks[i].ii = (int32_t *)malloc(rowCount[i]*sizeof(int32_t));
	rowBlocks[i].jj = (int32_t *)malloc(rowCount[i]*sizeof(int32_t));
	rowBlocks[i].vv = (double *)malloc(rowCount[i]*sizeof(double));
	rowBlocks[i].nnz = rowCount[i];
	
      } else{
	rowBlocks[i].nnz = 0;
      }
    }

    /* Pass again to store */
    for(int j=cStr; j<cStp; j++){
      const int k = jc[j+1] - jc[j];

      int colBlockId = colMap[j];
      for (unsigned int idx = 0; idx < k; idx++) {

	const unsigned int i = (ir[jc[j] + idx]);
	int rowBlockId = rowMap[i];

	rowBlocks[rowBlockId].ii[ptr[rowBlockId]] = i - rowStart[rowBlockId];
	rowBlocks[rowBlockId].jj[ptr[rowBlockId]] = j - colStart[colBlockId];
	rowBlocks[rowBlockId].vv[ptr[rowBlockId]] = vv[jc[j]+idx];
	
	ptr[rowBlockId]++;
      }
      
    }
    
 
    /* Scan to make sparse */
    cursor = 0;
    for(int i=0; i<nRowBlocks; i++){
      
      if (rowCount[i] > 0){
	//Ps->Pb[blkstr+cursor].ii = rowBlocks[i].ii;
	//Ps->Pb[blkstr+cursor].jj = rowBlocks[i].jj;
	//Ps->Pb[blkstr+cursor].vv = rowBlocks[i].vv;

	memcpy(Ps->Pb[blkstr+cursor].ii, rowBlocks[i].ii, rowCount[i] * sizeof(int32_t));
	memcpy(Ps->Pb[blkstr+cursor].jj, rowBlocks[i].jj, rowCount[i] * sizeof(int32_t));
	memcpy(Ps->Pb[blkstr+cursor].vv, rowBlocks[i].vv, rowCount[i] * sizeof(double));
	cursor++;
      }

    }

    
    for(int i=0; i<nRowBlocks; i++){

      if(rowBlocks[i].nnz>0){
	free( rowBlocks[i].ii );
	free( rowBlocks[i].jj );
	free( rowBlocks[i].vv );
      }

    }

    
    free(rowBlocks);
    free(rowCount);
    free(ptr);

  }


  /* finish the block construction */
  for(int j=0; j<nColBlocks; j++){
    
    const int k = Ps->jc[j+1] - Ps->jc[j];
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = Ps->jc[j] + idx;

      if( Ps->Pb[blk].nnz > 0 ){
	finishBlock(&Ps->Pb[blk]);
      }
      
    }
  }
  
}


/* Function that scans the intput CSC matrix and computes 
   the number of nnz blocks of the top level structure */   
int32_t count_nnz_from_top(top_lvl_csc   * const Ps,
			   int32_t const * const rowMap,
			   int32_t const * const colMap,
			   int32_t const * const colStart,
			   int32_t const * const colSizes,
			   int     const * const ir,
			   int     const * const jc,
			   int     const         nRowBlocks,
			   int     const         nColBlocks){

  Ps->jc = (int32_t*)calloc((nColBlocks+1), sizeof(int32_t));
  
  int blkcount = 0;
  for(int bi=0; bi<nColBlocks; bi++){

    int cursor = 0;
    int32_t *rowCount = (int32_t *)calloc(nColBlocks , sizeof(int32_t));
    int cStr = colStart[bi];
    int cStp = cStr + colSizes[bi];

    int colBlockId = colMap[bi];
    int blkstr = Ps->jc[bi];

    for(int j=cStr; j<cStp; j++){
      const int k = jc[j+1] - jc[j];


      for (unsigned int idx = 0; idx < k; idx++) {

	const unsigned int i = (ir[jc[j] + idx]);
	int rowBlockId = rowMap[i];
	rowCount[rowBlockId]++;
      }

    }

    for(int i=0; i<nRowBlocks; i++){
      if(rowCount[i] > 0){
	Ps->jc[bi+1]++;
	blkcount++;
      }
    }

  }

  /* scan prefix */
  int offset = 0;
  for(int i=0; i<nColBlocks+1; i++){
    int size = Ps->jc[i];
    Ps->jc[i] += offset;
    offset += size;
  }
 

  return blkcount;
}


/* Function used for debugging - computed the nnz of the top level matrix 
   from the dense top level */ 
int32_t count_nnz_toplvl(top_lvl_csc *Ps, int32_t *block_nnz, int nLeaves){

  Ps->jc = (int32_t*)calloc((nLeaves+1), sizeof(int32_t));
  int totalcount = 0;

  for(int j=0; j<nLeaves; j++){
    for(int i=0; i<nLeaves; i++){
      if(block_nnz[j * nLeaves + i] > 0){
	totalcount++;
	Ps->jc[j+1]++;
      }
    }
  }

  /* scan prefix */
  int offset = 0;
  for(int i=0; i<nLeaves+1; i++){
    int size = Ps->jc[i];
    Ps->jc[i] += offset;
    offset += size;
  }
  
  
  for(int i=0; i<8; i++){
    printf("scan: %d\n", Ps->jc[i]);
  }
  
  
  return totalcount;
    
}

/* Function used for debugging - Matches the dense and sparse top levels */ 
int32_t toplvl_checksum(top_lvl_csc *Ps, int32_t *block_nnz, int nLeaves, int N){

  int count = 0;
  int pass = 1;
  for(int j=0; j<nLeaves; j++){
    const int k = Ps->jc[j+1] - Ps->jc[j];
  
    for (unsigned int idx = 0; idx < k; idx++) {
    
      const unsigned int i = (Ps->ir[Ps->jc[j] + idx]);
      count += Ps->Pb[Ps->jc[j] + idx].nnz;

      pass &= (Ps->Pb[Ps->jc[j] + idx].nnz == block_nnz[j * nLeaves + i]);
      
    }
  }

  printf("MATRCH: %s\n", (pass) ? "PASS\n" : "FAIL" );
  return count;
}


void mergeLeafBoxes(int32_t *leafMap, int N, int pThresLow){

  int offset = 0;

  int prev_box = 0;
  int current_box = N;
  int next_box = 0;

  
  int leafCount_old = 0;
  int leafCount_new = 0;
  for(int i=0; i<N; i += offset){

    prev_box = current_box;
    current_box = leafMap[i];

   
    offset = current_box;
    if(i + offset < N){
      next_box = leafMap[i + offset];
    } else{
      next_box = N;
    }

    if(current_box < pThresLow){
      
      if(prev_box < next_box){ // merge with the previus box
	leafMap[i] = 0;
	leafMap[i - prev_box] += current_box;
	current_box = leafMap[i - prev_box];

      } else { // merge with the next box 
	leafMap[i] += next_box;
	leafMap[i + offset] = 0;
	current_box = leafMap[i];
	offset += next_box;

      }
      
    }

    leafCount_old++;
  }
  

}

void mergeLeafBoxes2(int32_t *leafMap, int N, int pThresLow){

  int offset = 0;

  int current_box = 0;
  int previus_box = 0;
  
  for(int i=0; i<N; i+=offset){

    int cpop = leafMap[current_box];

    offset = leafMap[i];
    if(cpop < pThresLow){ // Merge with next box
      int next_box = leafMap[i];
      leafMap[i] = 0;
      leafMap[current_box] += next_box;
    } else {
      previus_box = current_box;
      current_box = i;
    }
    
  }
  
  int cpop = leafMap[current_box];
  if( cpop < pThresLow ){ // Merge with previus box
    leafMap[current_box] = 0;
    leafMap[previus_box] += cpop;
  }

}

void updateColMap(int32_t *col2leaves, boxHelper *helper,
		  int N, int nBox){


  for(int i=0; i<nBox; i++){
    fillMap(&col2leaves[helper[i].str], helper[i].size, i);
  }

}

void updateColMap(int32_t *col2leaves, int32_t *boxStart, int32_t *boxSize,
		  int N, int nBox){


  for(int i=0; i<nBox; i++){

    fillMap(&col2leaves[boxStart[i]], boxSize[i], i);
  }

}


void extractPermutationVector(int32_t *p,
			      boxHelper *helper,
			      int32_t *boxStart,
			      int nBoxes, int N){

  int32_t *idx = (int32_t *)malloc(N * sizeof(int32_t));
  
  for(int i=0; i<N; i++){
    idx[i] = i;
  }

  for(int i=0; i<nBoxes; i++){
    memcpy( &p[helper[i].str], &idx[boxStart[helper[i].id]],
	    helper[i].size * sizeof(int32_t) );
  }

  free(idx);
}


int selectMerges(boxHelper *mergedBoxes, boxHelper *helper, int nBox, int pThresLow, int pThresHigh){

  /* Count the bins that must be merged */
  int mBox = 0;
  for(int i=0; i<nBox; i++){
    mBox += helper[i].shouldMerge;
  }


  /* Find the boxes that must be merged */
  int current = 0;
  for(int i=1; i<mBox; i++){
    
    if( helper[i].size < pThresLow ){ // Merge the box;
      
      if( helper[i].size + helper[current].size < pThresHigh ){ // Merge with the current
	helper[i].mergeWith = current;
	helper[current].size += helper[i].size; 
      } else { // Set a new current
	current = i;
      }
      
    } else {
      printf("PROBLEM entered area of large boxes\n");
    }
    
  }

  /* Merge the boxes */
  int boxCount = 0;
  int offset = 0;
  for(int i=0; i<nBox; i++){
    if(helper[i].mergeWith == -1){
      mergedBoxes[boxCount].size = helper[i].size;
      mergedBoxes[boxCount].str = offset;
      offset += mergedBoxes[boxCount].size;
      boxCount++;
    }
  }

  return boxCount;
  
}

int mergeBoxBase(int32_t *p,
		 int32_t *col2leaves,
		 int32_t *boxStart,
		 int32_t *boxSizes,
		 int numBoxes,
		 int N, int pThresLow, int pThresHigh);


void freeMergeBuffs(leafMergeBuffs *B){

  free( B->p );
  free( B->col2leaves );
  free( B->boxStart );
  free( B->boxSizes );
  free( B );
}

leafMergeBuffs *mergeBoxRec(node_t *box,
			   int32_t *p,
			   int32_t *col2leaves,
			   int32_t *boxStart,
			   int32_t *boxSizes,
			   int numBoxes,
			   int N,
			   int pThresLow,
			   int pThresHigh){

  leafMergeBuffs *leafBuffs = (leafMergeBuffs *)malloc(sizeof(leafMergeBuffs));

  
  if( box->numChild==0){

    /* peack up the part of the matrix 
       that belongs to the leaf */
    leafBuffs->nLeaves = 1;
    leafBuffs->numBoxes = 1;
    leafBuffs->nPoints = box->pop;
    leafBuffs->leavesStart = box->strParticle;
    /* Get the part of the permutation vector */
    leafBuffs->p = (int32_t *)malloc(box->pop * sizeof(int32_t));
    memcpy(leafBuffs->p, &p[box->strParticle], box->pop * sizeof(int32_t));
    /* Get the mapp of the columns */
    leafBuffs->col2leaves = (int32_t *)malloc(box->pop * sizeof(int32_t));
    memcpy(leafBuffs->col2leaves, &col2leaves[box->strParticle], box->pop * sizeof(int32_t));
    /* Get the start of the box */
    leafBuffs->boxStart  = (int32_t *)malloc(sizeof(int32_t));
    /* Get the size of the box */
    leafBuffs->boxSizes = (int32_t *)malloc(sizeof(int32_t));
    leafBuffs->boxSizes[0] = boxSizes[box->leafId];

      
  } else {


    leafMergeBuffs **childBuffs = (leafMergeBuffs **)malloc(box->numChild * sizeof(leafMergeBuffs*));
    
    node_t *current = box->first_child;
    for(int i=0; i<box->numChild; i++){
      childBuffs[i] = mergeBoxRec(current,
				  p,
				  col2leaves,
				  boxStart,
				  boxSizes,
				  numBoxes,
				  N, 
				  pThresLow,
				  pThresHigh);
      current = current->next_sibling;

    }
      
    /* Set up the merge buffers */ 
    leafBuffs->nLeaves = 0;
    leafBuffs->nPoints = 0;
    leafBuffs->numBoxes = 0;
    for(int i=0; i<box->numChild; i++){
      leafBuffs->nLeaves += childBuffs[i]->nLeaves;
      leafBuffs->nPoints += childBuffs[i]->nPoints;
      leafBuffs->numBoxes += childBuffs[i]->numBoxes;
    }

    /* Collect the parts of the permutation vector */
    leafBuffs->p = (int32_t *)malloc(leafBuffs->nPoints*sizeof(int32_t));

    /* Collect the parts of the column map */
    leafBuffs->col2leaves = (int32_t *)malloc(leafBuffs->nPoints * sizeof(int32_t));


    /* Colect the box pointers */
    leafBuffs->boxStart = (int32_t *)malloc(leafBuffs->nLeaves * sizeof(int32_t));


    /* Collect the box sizes */
    leafBuffs->boxSizes = (int32_t *)malloc(leafBuffs->nLeaves *sizeof(int32_t));

    int offsetPoints = 0;
    int offsetLeaves = 0; 
    for(int i=0; i<box->numChild; i++){
      memcpy(&leafBuffs->p[offsetPoints], childBuffs[i]->p, childBuffs[i]->nPoints*sizeof(int32_t));

      
      memcpy(&leafBuffs->col2leaves[offsetPoints], childBuffs[i]->col2leaves,
	     childBuffs[i]->nPoints*sizeof(int32_t));
      /*
      memcpy(&leafBuffs->boxStart[offsetLeaves], childBuffs[i]->boxStart,
	     childBuffs[i]->nLeaves*sizeof(int32_t));
      */
      memcpy(&leafBuffs->boxSizes[offsetLeaves], childBuffs[i]->boxSizes,
	     childBuffs[i]->nLeaves*sizeof(int32_t));
	
      offsetPoints += childBuffs[i]->nPoints;
      offsetLeaves += childBuffs[i]->nLeaves;
    }


    /* Free the buffers of the children */
    for(int i=0; i<box->numChild; i++){
      freeMergeBuffs( childBuffs[i] );
    }
    free(childBuffs);
    
    /* Scan prefix to find the start */
    int offset = 0;
    for(int i=0; i<leafBuffs->nLeaves; i++){
      leafBuffs->boxStart[i] = offset;
      offset += leafBuffs->boxSizes[i];
    }  

    if( leafBuffs->numBoxes < 16 ){
      int *pmerge = (int *)malloc( leafBuffs->nPoints * sizeof(int32_t) );
      leafBuffs->nLeaves = mergeBoxBase(pmerge,
					leafBuffs->col2leaves,
					leafBuffs->boxStart,
					leafBuffs->boxSizes,
					leafBuffs->nLeaves,
					leafBuffs->nPoints,
					pThresLow,
					pThresHigh);
      
      // ------- Fix the permutation vector
      int *pbuff = (int*)malloc(leafBuffs->nPoints * sizeof(int));
      for(int i=0; i<leafBuffs->nPoints; i++){
	pbuff[i] = leafBuffs->p[pmerge[i]];
      }
      memcpy(leafBuffs->p, pbuff, leafBuffs->nPoints * sizeof(int));
      
      free(pbuff);      
      free(pmerge);
    }
    
    
    
  }

  return leafBuffs;  

}

int mergeBoxBase(int32_t *p,
		 int32_t *col2leaves,
		 int32_t *boxStart,
		 int32_t *boxSizes,
		 int numBoxes,
		 int N, int pThresLow, int pThresHigh){
  
  boxHelper *helper = (boxHelper *)malloc(numBoxes * sizeof(boxHelper)); 
  boxHelper *mergedBoxes = (boxHelper *)malloc(numBoxes * sizeof(boxHelper));

  
  /* Check which boxes must be mearged */
  for(int i=0; i<numBoxes; i++){
    helper[i].shouldMerge = ( boxSizes[i] < pThresLow );
    helper[i].size = boxSizes[i];
    helper[i].id = i;
    helper[i].mergeWith = -1;
    mergedBoxes[i].size = 0;
  }

  /* Sort according to size */ 
  std::sort(helper, helper + numBoxes);

  
  /* Scan prefix of points */
  int offset = 0;
  for(int i=0; i<numBoxes; i++){
    helper[i].str = offset;
    offset += helper[i].size;
  }

  
  /* Generate permutation vector */
  extractPermutationVector(p,
			   helper,
			   boxStart,
			   numBoxes,
			   N);

  /* Merge boxes */
  int mergePop = 0;
  int finBinCount = selectMerges(mergedBoxes, helper, numBoxes, pThresLow, pThresHigh);
  
  for(int i=0; i<finBinCount; i++){
    mergePop += mergedBoxes[i].size;
  }
  

  /* Update the sizes */
  for(int i=0; i<numBoxes; i++){
    boxSizes[i] = mergedBoxes[i].size;
    boxStart[i] = mergedBoxes[i].str;
  }
  updateColMap(col2leaves, mergedBoxes, N, numBoxes);

  
  free( helper );
  free( mergedBoxes );
  
  return finBinCount;
}

/* Block sparse structure with sparse top level */
void form2levelSparseStruct(top_lvl_csc *Ptop,
                            int *pm, int *pn,
                            cs *Cp,
                            node_t *tree,
			    int32_t *pmerge,
                            int N,
                            int nnz, int pThres,
                            int pThres_low){
  
  // ----- memory allocations
  int32_t *leafMap    = (int32_t *)calloc( N, sizeof(int32_t) );
  int32_t *col2leaves = (int32_t *)calloc( N, sizeof(int32_t) );

  // ----- find size of each leaf
  createLeafMap(leafMap, tree);


#ifdef USE_PTHRES_LOW

  printf("merge enabled %d\n", pThres_low);
  int32_t *mergedLeafMap = (int32_t *)calloc(N,sizeof(int32_t));
  memcpy(mergedLeafMap, leafMap, N*sizeof(int32_t));
  
  mergeLeafBoxes2(mergedLeafMap, N, pThres_low);

  
#ifdef VERIFY
  int chsum = 0;
  for(int i=0; i<N; i++){
    chsum += mergedLeafMap[i];
    if(mergedLeafMap[i] < pThres_low && mergedLeafMap[i] > 0){
      printf("PROBLEM: %d, %d\n", i, mergedLeafMap[i]);
    }
  }
  printf("check: %d, N: %d\n", chsum, N);
  printf("Merging: %s\n", (chsum==N) ? "PASS" : "FAIL");
#endif
  
  free(leafMap);
  leafMap = mergedLeafMap;
  
#endif

  // ----- get a vector that says its element which block belongs to
  int32_t nLeaves = mapClo2Leaves(col2leaves, leafMap, N);

  // ----- allocate 2 vectors of size nLeaves
  int32_t *leafStart  = (int32_t *)calloc( nLeaves, sizeof(int32_t) );
  int32_t *leafSize   = (int32_t *)calloc( nLeaves, sizeof(int32_t) );


  // ------ Index of tree leaves
  updateLeafOrder(tree,
		  col2leaves,
		  leafStart,
		  leafSize);
  
  
  /* Linearize the leaves of the tree */
  updateLeafOrder(leafMap,
		  leafStart,
		  leafSize,
		  N);
  

#ifdef MERGE_LEAVES
  int *p = (int *)malloc(N * sizeof(int));
  for(int i=0; i<N; i++){
    p[i] = i;
  }
  leafMergeBuffs * mergeBuffs = mergeBoxRec(tree,
					    p,
					    col2leaves,
					    leafStart,
					    leafSize,
					    nLeaves,
					    N,
					    pThres_low,
					    pThres);

  printf("Number of leaves from tree: %d true leaves: %d\n", mergeBuffs->nLeaves, nLeaves);


  /* Copy the new order and free helping buffers */ 
  nLeaves = mergeBuffs->nLeaves;
  memcpy(pmerge, mergeBuffs->p, N*sizeof(int32_t));
  memcpy(leafStart, mergeBuffs->boxStart, nLeaves*sizeof(int32_t));
  memcpy(leafSize, mergeBuffs->boxSizes, nLeaves*sizeof(int32_t));
  /* Update the column map */
  updateColMap(col2leaves,
	       leafStart,
	       leafSize,
	       N, nLeaves);

  /* Clean the memory buffers used for the leaf merging */
  freeMergeBuffs( mergeBuffs );
  free( p );
  
  int *pm_inv = cs_pinv( pmerge, N );
  cs *Cp2 = cs_permute( Cp, pm_inv, pmerge, 1 );
  Cp = Cp2;

#endif


  
  /* Verify sum */
#ifdef VERIFY
  int lsum = 0;
  for(int i=0; i<nLeaves; i++){
    lsum += leafSize[i];
  }
  printf("Leaf sum: %d\n", lsum);
  printf("Leaf-sum: %s\n", (lsum == N) ? "PASS" : "FAIL");
#endif

  /* Count the nnz in each block */
  int32_t *block_nnz;
  
  /* Count the number of non-empty blocks at the top level */
  int nnz_top = count_nnz_from_top(Ptop,
				   col2leaves,
				   col2leaves,
				   leafStart,
				   leafSize,
				   Cp->i,
				   Cp->p,
				   nLeaves,
				   nLeaves);
  
  
  /* Form the data structure */

  printf("#### Number of source leaves: %d\n", nLeaves);
  printf("#### Number of target leaves: %d\n", nLeaves); 


  
  //printf("NNZ at the top: %d\n", nnz_top);

  /* Count the nnz per non empty block of the top level */
  count_block_nnz_sptop(Ptop,
			col2leaves,
			col2leaves,
			leafStart,
			leafSize,
			Cp->i,
			Cp->p,
			N,
			nLeaves,
			nnz_top);

  printf("Form the new matrix");
  /* Form the non-empty blocks of the top level (store and pack points) */
  formSparseBlockSparseTop(Ptop,
			   col2leaves,
			   col2leaves,
			   leafStart,
			   leafStart,
			   leafSize,
			   leafSize,
			   Cp->i,
			   Cp->p,
			   Cp->x,
			   N,
			   nLeaves,
			   nLeaves,
			   nnz);
  

  csc2csr_top(Ptop, nLeaves, nLeaves);    
  
  pm[0] = nLeaves;
  pn[0] = nLeaves;

  free( leafMap );
  free( leafStart );
  free( leafSize );
  free( col2leaves );
  // freeBlockSparseTopSparse(Ptop, nLeaves);

#ifdef MERGE_LEAVES 
  cs_spfree(Cp);
#endif
  
}


/* Free the 2nd level blocks */
void freeSparseBlock(sparseBlock *Ps){

  free( Ps->ju );
  free( Ps->li );
  free( Ps->ii );
  free( Ps->jj );
  free( Ps->vv );
  free( Ps->iu );
  
}



/* Free top level sparse structure */
void freeBlockSparseTopSparse(top_lvl_csc *Ps, int nColBlocks, bool rec){

  for(int j=0; j<nColBlocks; j++){

    const int k = Ps->jc[j+1] - Ps->jc[j];

    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = Ps->jc[j] + idx;

      sparseBlock *Bl = &Ps->Pb[blk];

      if (rec)
        freeSparseBlock(Bl);

    }
  }

  free( Ps->ir );
  free( Ps->jc );
  free( Ps->Pb );
  
}

/* travese the bssb structure in parallel 
   csc packing of the top level*/
void traverse_csc_top(double *F,
		      double *Y,
		      top_lvl_csc *Ps,
		      int nRowBlocks,
		      int nColBlocks,
		      int dim){

  cilk_for(int j=0; j<nColBlocks; j++){

    const int offCol = Ps->jc[j];
    
    const int k = Ps->jc[j+1] - offCol;

    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;
      
      sparseBlock *Bl = &Ps->Pb[blk];
      
      computeLeaf(Bl, F, Y, dim);
      
    }
    
  }


}


/* Top level caller function */
int CSC2BSSB(top_lvl_csc **Pt, int *p, cs *C,
             double *X,  int N, int d, int nnz,
             int maxLevel, int pThres, int pThresMin,
             int nworkers){

  // ----- GENERATE BSSB STRUCTURE

  // prepare permutation vector and permuted data points
  double   *Xt     = (double *)malloc(N*d*sizeof(double));
  uint32_t *pc     = (uint32_t *)malloc(N*sizeof(uint32_t));

  
  
  node_t *tree = NULL;
  treeOrder(&tree, Xt, pc, X, N, d, maxLevel, pThres, nworkers);
  
    
  // permute rows and columns of P to tree order
  for(int i=0; i<N; i++){
    p[i] = pc[i];
  }

  // permute CSC matrix
  int* p_inv = cs_pinv( p, N );
  cs *Cp     = cs_permute( C, p_inv, p, 1 );

  // form BSSB block
  Pt[0] = (top_lvl_csc*)malloc(sizeof(top_lvl_csc));
  int nLeaves2 = 0;
  int32_t *pmerge = (int32_t *)malloc(N * sizeof(int32_t));
  form2levelSparseStruct(Pt[0],
                         &nLeaves2, &nLeaves2,
                         Cp,
                         tree,
			 pmerge,
                         N,
                         nnz, pThres,
                         pThresMin);

#ifdef MERGE_LEAVES 
  /* Fix the permutation vector */
  //printf("\nFixing permutation vector according to the merge\n");
  int *pbuff = (int*)malloc(N * sizeof(int));
  for(int i=0; i<N; i++){
    pbuff[i] = p[pmerge[i]];
  }
  //printf("Copying the mpermutation vector");
  memcpy(p, pbuff, N * sizeof(int));
#endif

  printf("Freeing the memory\n");
  // free unecessary local variables
  free( Xt    );  
  free( p_inv );
  free( pc    );

  // free tree
  free_node(tree, 0);

  printf("Free Cp matrix\n");
  // free unused matrix
  cs_spfree(Cp);
  printf("Exitting CSC2BSSB\n");
  
  return nLeaves2;
  
}


/* Ordering function */
void treeOrderingWithMergedLeaves(node_t **root,
				  double *ordPoints,
				  uint32_t *pointId,
				  double *points,
				  int32_t N,
				  int32_t d,
				  int maxLevel,
				  int pThresHigh,
				  int pThresLow,
				  int nworkers){

  // --------- Build the tree
  treeOrder(root, ordPoints, pointId, points, N, d, maxLevel, pThresHigh, nworkers);
  
  node_t *tree = root[0];

  /* Merge the leaves and update the order */
  int32_t *leafMap    = (int32_t *)calloc( N, sizeof(int32_t) );
  int32_t *col2leaves = (int32_t *)calloc( N, sizeof(int32_t) );

  
  // ----- find size of each leaf
  createLeafMap(leafMap, tree);
  
  // ----- get a vector that says its element which block belongs to
  int32_t nLeaves = mapClo2Leaves(col2leaves, leafMap, N);

  // ----- allocate 2 vectors of size nLeaves
  int32_t *leafStart  = (int32_t *)calloc( nLeaves, sizeof(int32_t) );
  int32_t *leafSize   = (int32_t *)calloc( nLeaves, sizeof(int32_t) );


  // ------ Index of tree leaves
  updateLeafOrder(tree,
		  col2leaves,
		  leafStart,
		  leafSize);

  /* Linearize the leaves of the tree */
  updateLeafOrder(leafMap,
		  leafStart,
		  leafSize,
		  N);


  int *p = (int *)malloc(N * sizeof(int));
  for(int i=0; i<N; i++){
    p[i] = i;
  }
  
  leafMergeBuffs * mergeBuffs = mergeBoxRec(tree,
					    p,
					    col2leaves,
					    leafStart,
					    leafSize,
					    nLeaves,
					    N,
					    pThresLow,
					    pThresHigh);

  printf("Number of leaves from tree: %d true leaves: %d\n", mergeBuffs->nLeaves, nLeaves);

  /* Update the permutation vector */ 
  int *pbuff = (int *)malloc(N * sizeof(int));
  for(int i=0; i<N; i++){
    pbuff[i] = pointId[mergeBuffs->p[i]];
  }
  memcpy(pointId, pbuff, N * sizeof(int));
  
  free( pbuff );
  free( p );
  
  /* Clean memory */
  freeMergeBuffs( mergeBuffs );
  free( leafStart );
  free( leafSize );
  free( leafMap );
  free( col2leaves );
}
