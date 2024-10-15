/*!
  \file   gradient_descend.hpp
  \brief  Kullback-Leibler minimization routines

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef GRADIENT_DESCEND_HPP
#define GRADIENT_DESCEND_HPP

#include "types.hpp"
#include "utils.hpp"

//! Gradient descend for Kullback-Leibler minimization
/*!
*/
void kl_minimization(coord* Y,  //!< Embedding coordinates (output)
                     tsneparams param, //!< t-SNE parameters
                     sparse_matrix *P, //!< Sparse Matrix
                     double **timeInfo //!< [Optional] Timing information (output)
                     );

//! Compute the gradient vector
/*! 
 */
template <class dataPoint>
double compute_gradient(dataPoint *dy,
                        double *timeFrep,
                        double *timeFattr,
			tsneparams params,
			dataPoint *y,
      sparse_matrix *P,
                        double *timeInfo = nullptr);


#endif /* GRADIENT_DESCEND_HPP */
