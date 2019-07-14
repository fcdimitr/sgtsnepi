/*!
  \file   graph_rescaling.hpp
  \brief  Routines implementing lambda-based graph rescaling

  \author Dimitris Floros
  \date   2019-06-21
*/



#ifndef GRAPH_RESCALING_HPP
#define GRAPH_RESCALING_HPP

#include "sparsematrix.hpp"

//! Rescale given column-stochastic graph, using specified lambda parameter
/*!
*/
void lambdaRescaling( sparse_matrix P,        //!< Column-stocastic CSC matrix
                      double lambda,          //!< λ rescaling parameter
                      bool dist=false,        //!< [optional] Consider input as distance?
                      bool dropLeafEdge=false //!< [optional] Remove edges from leaf nodes?
                      );

#endif /* GRAPH_RESCALING_HPP */


