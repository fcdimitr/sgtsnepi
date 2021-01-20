/*!
  \file   pq.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-28
*/


#include <iostream>
#include <cstring>
#include "pq.hpp"

void pq( double       *       Fattr,
         double       * const Y,
         double const * const p_sp,
         matidx       *       ir,
         matidx       *       jc,
         int    const         n,   
         int    const         d) {
  
  std::memset( Fattr, 0, n*d * sizeof(double) );
  for (int j = 0; j < n; j++) {
  //for (unsigned int j = 0; j < n; j++) {

    double accum[3] = {0};
    double Yj[3];
    double Yi[3];

    const int k = jc[j+1] - jc[j];    /* number of nonzero elements of each column */

    std::memcpy( Yi, Y+(j*d), d*sizeof(double) );

    /* for each non zero element */
    for (unsigned int idx = 0; idx < k; idx++) {
      
      const unsigned int i = (ir[jc[j] + idx]);

      std::memcpy( Yj, Y+(i*d), d*sizeof(double) );

      /* distance computation */
      double dist = 0.0;
      for (int dd = 0; dd < d; dd++)
        dist += (Yj[dd] - Yi[dd]) * (Yj[dd] - Yi[dd]);

      /* P_{ij} \times Q_{ij} */
      const double p_times_q = p_sp[jc[j]+idx] / (1+dist);

      /* F_{attr}(i,j) */
      for (int dd = 0; dd < d; dd++) {
        Fattr[(i*d) + dd] += p_times_q * (Yj[dd] - Yi[dd]);
      }

    }
      
  }

}
