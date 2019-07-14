/*!
  \file   matrix_indexing.hpp
  \brief  List of macros for linear indexing of multidimensional arrays.

  \author Dimitris Floros
  \date   2019-04-28
*/


#ifndef _MATRIX_INDEXING_H_
#define _MATRIX_INDEXING_H_

// ==================== GENERAL MATRIX INDEXING MACROS

#define SUB2IND2DVECSTART(j,m)             ( (m) * (                                             (j) ) )
#define SUB2IND3DVECSTART(j,k,m,n)         ( (m) * ( (n) *                               (k)   + (j) ) )
#define SUB2IND4DVECSTART(j,k,l,m,n,o)     ( (m) * ( (n) * ( (o) *               (l)   + (k) ) + (j) ) )
#define SUB2IND5DVECSTART(j,k,l,a,m,n,o,p) ( (m) * ( (n) * ( (o) * ( (p) * (a) + (l) ) + (k) ) + (j) ) )

#define SUB2IND2D(i,j,m)               ( SUB2IND2DVECSTART(j,m)             + (i) )
#define SUB2IND3D(i,j,k,m,n)           ( SUB2IND3DVECSTART(j,k,m,n)         + (i) )
#define SUB2IND4D(i,j,k,l,m,n,o)       ( SUB2IND4DVECSTART(j,k,l,m,n,o)     + (i) )
#define SUB2IND5D(i,j,k,l,a,m,n,o,p)   ( SUB2IND5DVECSTART(j,k,l,a,m,n,o,p) + (i) )

#endif /* _MATRIX_INDEXING_H_ */
