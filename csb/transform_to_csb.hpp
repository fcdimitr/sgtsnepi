#include "benchmark_csb.hpp"

#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"

#include <utility> 

/**
 * Structure holding BSSB object, using CSB for each block
 *
 * NOTE: Currently not working optimally. The overhead for accessing
 * and processing each block appears to be more than it should.
 * 
 */
typedef struct{
  int *ir;                          // Start of every block row
  int *jc;                          // Index of the column blocks
  BiCsb<VALUETYPE, INDEXTYPE> **Sb; // Array of the sparse blocks CSB
  int *rowStart;                    // Row start of each block (global)
  int *colStart;                    // Col start of each block (global)
  int nRow;                         // Number of block rows
} bssb;

typedef struct{
  int rowStart;  // Offset of global row
  int colStart;  // Offset of global col
  int     Nrow;  // Number of rows in the block
  int     Ncol;  // Number of columns in the block
  int     *iu;   // Array of row indices with nonzero entries
  int     niu;   // Number of rows with nonzero entries
  int     *li;   // Number of nonzero entries in every row with nonzero entries
  int     *jj;   // Column index of the nonzero entries
  double  *vv;   // Value of the nonzero entries
  int     nnz;   // Number of non zeros in the block
} sparseBlock_CSR;

/**
 * Custom BSSB object, using our bottom level code
 * 
 */
typedef struct{
  int *ir;                      // Start of every block row
  int *jc;                      // Index of the column blocks
  sparseBlock_CSR *Sb;          // Array of the custom sparse blocks
  int nRow;                     // Number of block rows
} bssb_custom;

/* travese the bssb structure in parallel */
void traverse_bssb(double *F,
                   double *Y,
                   bssb   BSSB,
                   int dim){

  cilk_for(int i=0; i<BSSB.nRow; i++){

    const int offCol = BSSB.ir[i];
    
    const int k = BSSB.ir[i+1] - offCol;

    // printf("\n New for \n\n");
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;

      int j = BSSB.jc[blk];
      
      BiCsb<VALUETYPE, INDEXTYPE> *Bl = BSSB.Sb[blk];

      int rowStart = BSSB.rowStart[blk];
      int colStart = BSSB.colStart[blk];
      
      // printf("Blk (%d,%d): offset (%d,%d)\n", i, j,
             // rowStart, colStart);
      typedef PTSR<VALUETYPE,VALUETYPE> PTDD;
      bicsb_gespmv<PTDD>(Bl[0], &Y[colStart], &F[rowStart]);
      // computeLeaf(Bl, F, Y, dim);
      
    }
    
  }


}

/* travese the bssb structure in parallel */
void traverse_bssb_tsne(double *F,
			double *Y,
			bssb   BSSB,
			int dim){

  cilk_for(int i=0; i<BSSB.nRow; i++){

    const int offCol = BSSB.ir[i];
    
    const int k = BSSB.ir[i+1] - offCol;

    // printf("\n New for \n\n");
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;

      int j = BSSB.jc[blk];
      
      BiCsb<VALUETYPE, INDEXTYPE> *Bl = BSSB.Sb[blk];

      int rowStart = BSSB.rowStart[blk];
      int colStart = BSSB.colStart[blk];
      
      // printf("Blk (%d,%d): offset (%d,%d)\n", i, j,
             // rowStart, colStart);
      typedef PTSR<VALUETYPE,VALUETYPE> PTDD;
      bicsb_tsne<PTDD>(Bl[0], &Y[dim*rowStart], &Y[dim*colStart], &F[dim*rowStart]);

      // computeLeaf(Bl, F, Y, dim);
      
    }
    
  }


}


void csc2csb_bot(bssb *BSSB, top_lvl_csc *BSSB_CSR, int nCol, int nRow,
                 int workers, int csbBeta){

  int *csr_jr = BSSB_CSR->jc;
  int *csr_ic = BSSB_CSR->ir;
  sparseBlock *csr_sb = BSSB_CSR->Pb;

  int nnz = csr_jr[nCol];

  // PREPARE NEW OBJECT BSSB
  
  BSSB->ir = csr_jr;
  BSSB->jc = csr_ic;
  BSSB->Sb = (BiCsb<VALUETYPE, INDEXTYPE> **)
    malloc(nnz*sizeof(BiCsb<VALUETYPE, INDEXTYPE>*));
  BSSB->rowStart = (int *)malloc(nnz*sizeof(int));
  BSSB->colStart = (int *)malloc(nnz*sizeof(int)); 
  BSSB->nRow = nRow;

  // PASS TRHOUGH OLD STRUCT TO CONSTRUCT NEW OBJECT
  
  for(int i=0; i<nRow; i++){

    const int offCol = csr_jr[i];
    
    const int k = csr_jr[i+1] - offCol;
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;
      int j   = csr_ic[blk];
      sparseBlock *Bl = &csr_sb[blk];

      int nnzBlk = Bl->nnz;

      BSSB->rowStart[blk] = Bl->row;
      BSSB->colStart[blk] = Bl->col;
      
      INDEXTYPE * rowindices = new INDEXTYPE[nnzBlk];
      INDEXTYPE * colindices = new INDEXTYPE[nnzBlk];
      VALUETYPE * vals       = new VALUETYPE[nnzBlk];
      
      int32_t nnzCol = Bl->nuj;
      double     *vv = Bl->vv;
      int32_t    *ju = Bl->ju;
      int32_t    *ii = Bl->ii;
      int32_t    *li = Bl->li;
      int32_t      m = Bl->Nrow;
      int32_t      n = Bl->Ncol;
      int32_t     ss = 0;

      printf("B[%d,%d] = Row: %d Col:%d | %dx%d nnz: %d\n", i, j, Bl->row, Bl->col, m,n, nnzBlk);

      int itemIter = 0;
      
      for (uint32_t j_blk = 0; j_blk < nnzCol; j_blk++) {

        const int32_t k_blk = li[j_blk];
        // printf("  k_blk = %d\n", k_blk);
        for (uint32_t idx_blk = 0; idx_blk < k_blk; idx_blk++) {
          
          const uint32_t i_blk = (ii[ss + idx_blk]);
          // printf("  Bp[%d,%d] = %.2g\n", i_blk, ju[j_blk], vv[ss+idx_blk]);
          rowindices[itemIter] = i_blk;
          colindices[itemIter] = ju[j_blk];
          vals[itemIter]       = vv[ss+idx_blk];
          itemIter++;
        }

        ss += k_blk;   
        
      }

      Csc<double, uint32_t> * csc;
      csc = new Csc<VALUETYPE, INDEXTYPE>(rowindices, colindices, vals, nnzBlk, m, n);


      float csbBetaNew = 0.0;
      if (csbBeta == 0){
        csbBetaNew = floor( (double) log2(max(m,n)));
      }
      
      BSSB->Sb[blk] = new BiCsb<VALUETYPE, INDEXTYPE>(*csc, workers,
                                                      (int)csbBetaNew);

      printf("Blk stats: nRow = %d | nCol = %d\n",
             BSSB->Sb[blk]->getNbr(),
             BSSB->Sb[blk]->getNbc());
      
      freeSparseBlock(Bl);
      
      delete [] rowindices;
      delete [] colindices;
      delete [] vals;
      
      // computeLeaf(Bl, F, Y, dim);
      
    }
    
  }

}




void csc2csr_bot(bssb_custom *BSSB, top_lvl_csc *BSSB_CSR, int nCol, int nRow,
                 int workers, int csbBeta, bool freePrev){

  int *csr_jr = BSSB_CSR->jc;
  int *csr_ic = BSSB_CSR->ir;
  sparseBlock *csr_sb = BSSB_CSR->Pb;

  int nnz = csr_jr[nCol];

  // PREPARE NEW OBJECT BSSB
  
  BSSB->ir = csr_jr;
  BSSB->jc = csr_ic;
  BSSB->Sb = (sparseBlock_CSR *)
    malloc(nnz*sizeof(sparseBlock_CSR));
  BSSB->nRow = nRow;

  int total_nnz = 0;
  int total_niu = 0;
  
  // PASS TRHOUGH OLD STRUCT TO CONSTRUCT NEW OBJECT
  
  for(int i=0; i<nRow; i++){

    const int offCol = csr_jr[i];
    
    const int k = csr_jr[i+1] - offCol;
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;
      int j   = csr_ic[blk];
      sparseBlock *Bl = &csr_sb[blk];

      // create new sparseBlock CSR
      sparseBlock_CSR *Sb_temp = &BSSB->Sb[blk];
      
      int nnzBlk = Bl->nnz;

      // pass scalar values
      Sb_temp->rowStart = Bl->row;
      Sb_temp->colStart = Bl->col;
      Sb_temp->Nrow     = Bl->Nrow;
      Sb_temp->Ncol     = Bl->Ncol;
      Sb_temp->nnz      = Bl->nnz;
      
      // allocate sparse block elements
      Sb_temp->vv = (double *)malloc(Sb_temp->nnz*sizeof(double));
      Sb_temp->jj = (int *)malloc(Sb_temp->nnz*sizeof(double));

      // prepare array of vectors
      std::vector< std::pair<int,double> > **rows =
        (std::vector< std::pair<int,double> > **)
        malloc(Sb_temp->Nrow*sizeof(std::vector< std::pair<int,double> >*));

      // allocate vectors
      for (int iVec = 0; iVec<Sb_temp->Nrow; iVec++){

        rows[iVec] = new std::vector< std::pair<int,double> >();

      }

      // transform CSC to CSR
      int32_t nnzCol = Bl->nuj;
      int   itemIter = 0;
      int32_t     ss = 0;
            
      for (uint32_t j_blk = 0; j_blk < nnzCol; j_blk++) {

        const int32_t k_blk = Bl->li[j_blk];
        // printf("  k_blk = %d\n", k_blk);
        for (uint32_t idx_blk = 0; idx_blk < k_blk; idx_blk++) {
          
          uint32_t i_blk = (Bl->ii[ss + idx_blk]);
          double   v_blk = (Bl->vv[ss + idx_blk]);
          std::pair<int,double> newPair(Bl->ju[j_blk], v_blk);
          rows[i_blk]->push_back( newPair );
          
        }

        ss += k_blk;
        
      }

      int niu = 0;

      // find number of nnz rows
      for (int iVec = 0; iVec<Sb_temp->Nrow; iVec++){
        if (rows[iVec]->size() > 0) niu++;        
      }

      Sb_temp->niu = niu;

      // update total values
      total_nnz += Sb_temp->nnz;
      total_niu += Sb_temp->niu;
      
      // allocate remaining vectors
      Sb_temp->iu = (int *)malloc(niu*sizeof(int));
      Sb_temp->li = (int *)malloc(niu*sizeof(int));

      int iterRow = 0;
      int iterNnz = 0;
      
      // fill vectors
      for (int iVec = 0; iVec<Sb_temp->Nrow; iVec++){
        
        if (rows[iVec]->size() > 0){ // if not empty

          Sb_temp->iu[iterRow] = iVec;
          Sb_temp->li[iterRow] = rows[iVec]->size();
          
          for (int jVec = 0; jVec<rows[iVec]->size(); jVec++){

            Sb_temp->jj[iterNnz] = rows[iVec]->at(jVec).first;
            Sb_temp->vv[iterNnz] = rows[iVec]->at(jVec).second;

            // FILE *f_i = fopen( "csr_con_i.bin", "ab" );
            // FILE *f_j = fopen( "csr_con_j.bin", "ab" );
            // FILE *f_v = fopen( "csr_con_v.bin", "ab" );

            // int    i_bin = iVec+Sb_temp->rowStart;
            // int    j_bin = Sb_temp->jj[iterNnz]+Sb_temp->colStart;
            // double v_bin = Sb_temp->vv[iterNnz];
      
            // fwrite( &i_bin, sizeof(i_bin), 1, f_i );
            // fwrite( &j_bin, sizeof(j_bin), 1, f_j );
            // fwrite( &v_bin, sizeof(v_bin), 1, f_v );

            // fclose( f_i ); fclose( f_j ); fclose( f_v );
            
            iterNnz++;
              
          }

          iterRow++;
          
        }
        
      }

      // printf("Block (%d,%d) row: [%d,%d] col:[%d,%d] nnz=%d gold=%d\n",
      //        i, j, 
      //        Sb_temp->rowStart, Sb_temp->rowStart + Sb_temp->Nrow,
      //        Sb_temp->colStart, Sb_temp->colStart + Sb_temp->Ncol,
      //        iterNnz, nnzBlk);
      
      // de-allocate vectors
      for (int iVec = 0; iVec<Sb_temp->Nrow; iVec++){

        delete rows[iVec];

      }

      free(rows);
      
      if (freePrev)
        freeSparseBlock(Bl);
      
    } // finish block row
    
  } // finish block col

  printf("total nnz: %d, total niu:%d\n", total_nnz, total_niu);

  int *global_iu    = (int    *)malloc( total_niu*sizeof(int)    );
  int *global_li    = (int    *)malloc( total_niu*sizeof(int)    );
  int *global_jj    = (int    *)malloc( total_nnz*sizeof(int)    );
  double *global_vv = (double *)malloc( total_nnz*sizeof(double) );

  // traverse and update pointers

  total_niu = 0;
  total_nnz = 0;
  
  for(int i=0; i<BSSB->nRow; i++){

    const int offRow = BSSB->ir[i];
    
    const int k = BSSB->ir[i+1] - offRow;

    // printf("\n New for \n\n");
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offRow + idx;

      int j = BSSB->jc[blk];
      
      sparseBlock_CSR *Bl = &BSSB->Sb[blk];

      for ( int iter = 0; iter < Bl->nnz; iter++ ){

        global_jj[total_nnz + iter] = Bl->jj[iter];
        global_vv[total_nnz + iter] = Bl->vv[iter];

      }

      for ( int iter = 0; iter < Bl->niu; iter++ ){

        global_iu[total_niu + iter] = Bl->iu[iter];
        global_li[total_niu + iter] = Bl->li[iter];

      }

      free( Bl->jj );
      free( Bl->vv );
      free( Bl->iu );
      free( Bl->li );

      Bl->jj = &global_jj[total_nnz];
      Bl->vv = &global_vv[total_nnz];

      Bl->iu = &global_iu[total_niu];
      Bl->li = &global_li[total_niu];
      
      total_niu += Bl->niu;
      total_nnz += Bl->nnz;

    }
    
  }
  
}


/* PQ Kernel */
void computeLeafCSRtsne(sparseBlock_CSR* Ps, double* F, double* Y, int dim){

  double   *vv = Ps->vv;
  int      *iu = Ps->iu;
  int      *jj = Ps->jj;
  int      *li = Ps->li;
  int       m = Ps->Nrow;
  int       n = Ps->Ncol;

  int nn = Ps->niu;

  int ss = 0;

  int R = Ps->rowStart;
  int C = Ps->colStart;
	       
  double* Y0i = &Y[R * dim];
  double* Y0j = &Y[C * dim];

  double* F0i = &F[R * dim];

  for (int i = 0; i < nn; i++) {

    double accum[3] = {0};
    double Ftemp[3] = {0};
    double Yj[3] = {0};
    double Yi[3] = {0};

    const int k = li[i];    /* number of nonzero elements of each row */

    
    Yi[:] = Y0i[ iu[i]*dim + 0:dim ];
    accum[:] = 0;

    /* for each non zero element */
    for (int idx = 0; idx < k; idx++) {

      const int j = (jj[ss + idx]);

      Yj[:] = Y0j[ j * dim + 0:dim ];

      /* distance computation */
      double dist = __sec_reduce_add( (Yi[:] - Yj[:])*(Yi[:] - Yj[:]) );

      // FILE *f_i = fopen( "csr_i.bin", "ab" );
      // FILE *f_j = fopen( "csr_j.bin", "ab" );
      // FILE *f_v = fopen( "csr_v.bin", "ab" );

      // int    i_bin = iu[i] + R;
      // int    j_bin = j + C;
      // double v_bin = vv[ss+idx];
      
      // fwrite( &i_bin, sizeof(i_bin), 1, f_i );
      // fwrite( &j_bin, sizeof(j_bin), 1, f_j );
      // fwrite( &v_bin, sizeof(v_bin), 1, f_v );

      // fclose( f_i ); fclose( f_j ); fclose( f_v );
      
      /* P_{ij} \times Q_{ij} */
      double p_times_q = vv[ss+idx] / (1+dist);

      Ftemp[:] = p_times_q * ( Yi[:] - Yj[:] );

      /* F_{attr}(i,j) */
      accum[:] += Ftemp[:];
    }

    F0i[iu[i]*dim + 0:dim] += accum[:];
    ss += k;

  }
    
    
}


/* travese the bssb structure in parallel */
void traverse_bssb_csr_tsne(double      *F,
                            double      *Y,
                            bssb_custom BSSB,
                            int         dim,
                            int         nworkers){


#pragma cilk grainsize 1
  cilk_for (int thr = 0; thr < nworkers; thr++){
    
    for(int i=thr; i<BSSB.nRow; i+=nworkers){

      const int offCol = BSSB.ir[i];
    
      const int k = BSSB.ir[i+1] - offCol;

      // printf("\n New for \n\n");

      for (unsigned int idx = 0; idx < k; idx++) {
        
        int blk = offCol + idx;

        int j = BSSB.jc[blk];
      
        sparseBlock_CSR Bl = BSSB.Sb[blk];

      
      
        computeLeafCSRtsne(&Bl, F, Y, dim);
      
      } // for (idx, nColumns)
    
    } // for (i, nBlockRow)

  } // cilk_for (thr, nworkers)

}





/* PQ Kernel */
void computeLeafCSRspmv(sparseBlock_CSR* Ps, double* F, double* Y){

  double   *vv = Ps->vv;
  int      *iu = Ps->iu;
  int      *jj = Ps->jj;
  int      *li = Ps->li;
  int       m = Ps->Nrow;
  int       n = Ps->Ncol;

  int nn = Ps->niu;
  int ss = 0;

  int R = Ps->rowStart;
  int C = Ps->colStart;
	       
  double* Y0j = &Y[C];

  double* F0i = &F[R];

  for (int i = 0; i < nn; i++) {

    double accum = 0;
    
    // nnz of this row
    const int k = li[i];
    
    /* for each non zero element in row */
    for (int idx = 0; idx < k; idx++) {

      // get column index
      const int j = (jj[ss + idx]);

      // update results
      F0i[iu[i]] += vv[ ss+idx ] * Y0j[ j ];

    }

    // increase iterator by k
    ss += k;

  }
    
    
}


/* travese the bssb structure in parallel */
void traverse_bssb_csr_spmv(double      *F,
                            double      *Y,
                            bssb_custom BSSB){

  cilk_for(int i=0; i<BSSB.nRow; i++){

    const int offCol = BSSB.ir[i];
    
    const int k = BSSB.ir[i+1] - offCol;

    // printf("\n New for \n\n");
    
    for (unsigned int idx = 0; idx < k; idx++) {

      int blk = offCol + idx;

      int j = BSSB.jc[blk];
      
      sparseBlock_CSR Bl = BSSB.Sb[blk]; 
      
      computeLeafCSRspmv(&Bl, F, Y);
      
    }
    
  }


}

