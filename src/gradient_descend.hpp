/*!
  \file   gradient_descend.hpp
  \brief  Kullback-Leibler minimization routines

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef GRADIENT_DESCEND_HPP
#define GRADIENT_DESCEND_HPP

#include "types.hpp"
#include "../csb/csb_wrapper.hpp"
#include "utils.hpp"

//! Gradient descend for Kullback-Leibler minimization
/*!
*/
void kl_minimization(coord* Y,  //!< Embedding coordinates (output)
                     tsneparams param, //!< t-SNE parameters
                     BiCsb<matval, matidx> *csb, //!< CSB object
                     double **timeInfo //!< [Optional] Timing information (output)
                     );


#endif /* GRADIENT_DESCEND_HPP */
