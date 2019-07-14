/*!
  \file   pq.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-28
*/


#include <iostream>
#include "pq.hpp"

void pq( double       *       Fattr,
         double       * const Y,
         double const * const p_sp,
         matidx       *       ir,
         matidx       *       jc,
         int    const         n,   
         int    const         d) {
  
  Fattr[0:n*d] = 0; 
  for (int j = 0; j < n; j++) {
  //for (unsigned int j = 0; j < n; j++) {

    double accum[3] = {0};
    double Yj[3];
    double Yi[3];
    double Ftemp[3];
  
    const int k = jc[j+1] - jc[j];    /* number of nonzero elements of each column */

    Yi[0:d] = Y[ (j*d) + 0:d ];
    
    /* for each non zero element */
    for (unsigned int idx = 0; idx < k; idx++) {
      
      const unsigned int i = (ir[jc[j] + idx]);

      Yj[0:d] = Y[ (i*d) + 0:d ];

      /* distance computation */
      double dist = __sec_reduce_add( (Yj[0:d] - Yi[0:d])*(Yj[0:d] - Yi[0:d]) );

      /* P_{ij} \times Q_{ij} */
      const double p_times_q = p_sp[jc[j]+idx] / (1+dist);

      Ftemp[0:d] = p_times_q * ( Yj[0:d] - Yi[0:d] );

      /* F_{attr}(i,j) */
      Fattr[ (i*d) + 0:d ] += Ftemp[0:d];
      
    }
      
  }

}
