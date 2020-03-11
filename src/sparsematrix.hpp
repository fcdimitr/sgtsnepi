/*!
  \file   sparsematrix.hpp
  \brief  Basic sparse matrix routines.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include "sgtsne.hpp"
#include "utils.hpp"
#include <cstdlib>

//! Free allocated memory for CSC sparse matrix storage
/*!

  \param P      Sparse matrix in CSC format.
*/
void free_sparse_matrix(sparse_matrix * P);


//! Transform input matrix to stochastic
/*!

  \param P      Sparse matrix in CSC format.
  \return       Number of nodes already stochastic
*/
uint32_t makeStochastic(sparse_matrix P);
  
//! Print sparse matrix P (only print size if too large)
/*!

  \param P      Sparse matrix in CSC format.
*/
void printSparseMatrix(sparse_matrix P);

//! Symmetrize matrix P
/*!

  \param[in,out] P      Sparse matrix in CSC format.
*/
void symmetrizeMatrix( sparse_matrix *P );

//! Check if P has symmetric pattern
/*!

  \param[in,out] P      Sparse matrix in CSC format.
*/
bool isSymPattern( sparse_matrix *P );

//! Check if P is symmetric
/*!

  \param[in,out] P      Sparse matrix in CSC format.
*/
bool isSymValues( sparse_matrix *P );

//! Symmetrize matrix P (already symmetric pattern)
/*!

  \param[in,out] P      Sparse matrix in CSC format.
*/
void symmetrizeMatrixWithSymPat( sparse_matrix *P );

//! Permute matrix P
/*!

  \param[in,out] P      Sparse matrix in CSC format.
  \param perm           Permutation vector
  \param iperm          Inverse permutation vector
*/
void permuteMatrix( sparse_matrix *P, int *perm, int *iperm );

#endif /* SPARSEMATRIX_HPP */
