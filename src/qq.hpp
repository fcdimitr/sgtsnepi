#ifndef QQ_HPP
#define QQ_HPP

/*!
  \file   qq.hpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-20
*/

#include "types.hpp"

extern "C" {

coord computeFrepulsive_exact(coord * frep,
                              coord * pointsX,
                              int N,
                              int d);

coord computeFrepulsive_interp(coord * Frep,
                               coord * yin,
                               int n,
                               int d,
                               double h,
                               int np,
                               double *timeInfo = nullptr);

}

#endif /* QQ_HPP */
