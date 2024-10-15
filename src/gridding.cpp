/*!
  \file   gridding.cpp
  \brief  

  <long description>

  \author Dimitris Floros
  \date   2019-06-20
*/


#include <iostream>
#include <string>
#include <cmath>
 #include "cilk.hpp"
#include "matrix_indexing.hpp"

#define LAGRANGE_INTERPOLATION

#define y(i,j)      y[ SUB2IND2D((i),(j),nDim) ]
#define q(i,j)      q[ SUB2IND2D((i),(j),nVec) ]
#define Phi(i,j)  Phi[ SUB2IND2D((i),(j),nVec) ]

#define V1(i,j,k)       V[ SUB2IND3D((i),(j),(k),ng,nVec) ]
#define V2(i,j,k,l)     V[ SUB2IND4D((i),(j),(k),(l),ng,ng,nVec) ]
#define V3(i,j,k,l,m)   V[ SUB2IND5D((i),(j),(k),(l),(m),ng,ng,ng,nVec) ]

#ifdef LAGRANGE_INTERPOLATION

__inline__
double g1(double d){
  return   0.5 * d*d*d - 1.0 * d*d - 0.5   * d + 1;
}
  
__inline__
double g2(double d){
  double cc = 1.0/6.0;
  return -cc * d*d*d + 1.0 * d*d - 11*cc * d + 1;
}

#else

__inline__
double g1(double d){
  return  1.5 * d*d*d - 2.5 * d*d         + 1;
}
  
__inline__
double g2(double d){
  return -0.5 * d*d*d + 2.5 * d*d - 4 * d + 2;
}


#endif


void s2g1d( double * V,
            double * y,
            double * q,
            uint32_t ng,
            uint32_t np,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  #ifdef OPENCILK
  #pragma cilk grainsize 1
  #endif // OPENCILK
  CILK_FOR (uint32_t pid = 0; pid<np; pid++){

    double v1[4];
    
    for (uint32_t i = pid; i<nPts; i+=np){

      uint32_t f1;
      double d;
      
      f1 = (uint32_t) floor( y(0,i) );
      d = y(0,i) - (double) f1;

      v1[0] = g2(1+d);
      v1[1] = g1(  d);
      v1[2] = g1(1-d);
      v1[3] = g2(2-d);

      for (uint32_t j = 0; j<nVec; j++){

        double qv = q(j,i);
        
        for (uint32_t idx1 = 0; idx1<4; idx1++){
          V1(f1+idx1, j, pid) += qv * v1[idx1];
          
        }
        
      }
      
    } // (i)

  } // (pid)
  
  
}


void s2g1drb( double * V,
              double * y,
              double * q,
              uint32_t * ib,
              uint32_t * cb,
              uint32_t ng,
              uint32_t np,
              uint32_t nPts,
              uint32_t nDim,
              uint32_t nVec) {

  for (uint32_t s = 0; s < 2; s++ ) { // red-black sync

    CILK_FOR (uint32_t idual = 0; idual < (ng-3) ; idual += 6) { // coarse-grid

      for (uint32_t ifine = 0 ; ifine < 3 ; ifine++ ) { // fine-grid

        // get index of current grid box
        uint32_t i = 3*s+idual + ifine;

        // if above boundaries, break
        if (i > ng-4) break;

        // loop through all points inside box
        for (uint32_t k = 0; k < cb[i]; k++) {

          uint32_t f1;
          double d;
          double v1[4];
          
          f1 = (uint32_t) floor( y(0,ib[i]+k) );
          d = y(0,ib[i]+k)  - (double) f1;

          v1[0] = g2(1+d);
          v1[1] = g1(  d);
          v1[2] = g1(1-d);
          v1[3] = g2(2-d);

          for (uint32_t j = 0; j<nVec; j++){

            double qv = q(j,ib[i]+k);
        
            for (uint32_t idx1 = 0; idx1<4; idx1++){
              V1(f1+idx1, j, 0) += qv * v1[idx1];
          
            } // (idx1)
        
          } // (j)
          
        } // (k)
        
      } // (ifine)
      
    } // (idual)
    
  } // (s)

}


void s2g2d( double * V,
            double * y,
            double * q,
            uint32_t ng,
            uint32_t np,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  #ifdef OPENCILK
  #pragma cilk grainsize 1
  #endif // OPENCILK
  CILK_FOR (uint32_t pid = 0; pid<np; pid++){

    double v1[4];
    double v2[4];
    
    for (uint32_t i = pid; i<nPts; i+=np){

      uint32_t f1, f2;
      double d;
      
      f1 = (uint32_t) floor( y(0,i) );
      d = y(0,i) - (double) f1;

      v1[0] = g2(1+d);
      v1[1] = g1(  d);
      v1[2] = g1(1-d);
      v1[3] = g2(2-d);

      f2 = (uint32_t) floor( y(1,i) );
      d = y(1,i) - (double) f2;
      
      v2[0] = g2(1+d);
      v2[1] = g1(  d);
      v2[2] = g1(1-d);
      v2[3] = g2(2-d);

      for (uint32_t j = 0; j<nVec; j++){

        for (uint32_t idx2 = 0; idx2<4; idx2++){

          double qv2 = q(j,i) * v2[idx2];
          
          for (uint32_t idx1 = 0; idx1<4; idx1++){
            V2(f1+idx1, f2+idx2, j, pid) += qv2 * v1[idx1];
            
          }

        }
        
      }
      
    } // (i)

  } // (pid)
  
  
}


void s2g2drb( double * V,
              double * y,
              double * q,
              uint32_t * ib,
              uint32_t * cb,
              uint32_t ng,
              uint32_t np,
              uint32_t nPts,
              uint32_t nDim,
              uint32_t nVec) {

  for (uint32_t s = 0; s < 2; s++ ) { // red-black sync

    CILK_FOR (uint32_t idual = 0; idual < (ng-3) ; idual += 6) { // coarse-grid

      for (uint32_t ifine = 0 ; ifine < 3 ; ifine++ ) { // fine-grid

        // get index of current grid box
        uint32_t i = 3*s+idual + ifine;

        // if above boundaries, break
        if (i > ng-4) break;

        // loop through all points inside box
        for (uint32_t k = 0; k < cb[i]; k++) {

          uint32_t f1, f2;
          double d;
          double v1[4], v2[4];
          
          f1 = (uint32_t) floor( y(0,ib[i]+k) );
          d = y(0,ib[i]+k)  - (double) f1;

          v1[0] = g2(1+d);
          v1[1] = g1(  d);
          v1[2] = g1(1-d);
          v1[3] = g2(2-d);

          f2 = (uint32_t) floor( y(1,ib[i]+k) );
          d = y(1,ib[i]+k)  - (double) f2;

          v2[0] = g2(1+d);
          v2[1] = g1(  d);
          v2[2] = g1(1-d);
          v2[3] = g2(2-d);

          for (uint32_t j = 0; j<nVec; j++){

            for (uint32_t idx2 = 0; idx2<4; idx2++){

              double qv2 = q(j,ib[i]+k) * v2[idx2];
        
              for (uint32_t idx1 = 0; idx1<4; idx1++){
                V2(f1+idx1, f2+idx2, j, 0) += qv2 * v1[idx1];
          
              } // (idx1)

            } // (idx2)
              
          } // (j)
          
        } // (k)
        
      } // (ifine)
      
    } // (idual)
    
  } // (s)

}




void s2g3d( double * V,
            double * y,
            double * q,
            uint32_t ng,
            uint32_t np,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  #ifdef OPENCILK
  #pragma cilk grainsize 1
  #endif // OPENCILK
  CILK_FOR (uint32_t pid = 0; pid<np; pid++){

    double v1[4];
    double v2[4];
    double v3[4];
    
    for (uint32_t i = pid; i<nPts; i+=np){

      uint32_t f1, f2, f3;
      double d;
      
      f1 = (uint32_t) floor( y(0,i) );
      d = y(0,i) - (double) f1;

      v1[0] = g2(1+d);
      v1[1] = g1(  d);
      v1[2] = g1(1-d);
      v1[3] = g2(2-d);

      f2 = (uint32_t) floor( y(1,i) );
      d = y(1,i) - (double) f2;
      
      v2[0] = g2(1+d);
      v2[1] = g1(  d);
      v2[2] = g1(1-d);
      v2[3] = g2(2-d);

      f3 = (uint32_t) floor( y(2,i) );
      d = y(2,i) - (double) f3;
      
      v3[0] = g2(1+d);
      v3[1] = g1(  d);
      v3[2] = g1(1-d);
      v3[3] = g2(2-d);

      for (uint32_t j = 0; j<nVec; j++){

        for (uint32_t idx3 = 0; idx3<4; idx3++){
        
          for (uint32_t idx2 = 0; idx2<4; idx2++){

            double qv2v3 = q(j,i) * v2[idx2] * v3[idx3];
          
            for (uint32_t idx1 = 0; idx1<4; idx1++){
              V3(f1+idx1, f2+idx2, f3+idx3, j, pid) += qv2v3 * v1[idx1];
            
            }

          }
        
        }

      }
        
    } // (i)

  } // (pid)
  
  
}


void s2g3drb( double * V,
              double * y,
              double * q,
              uint32_t * ib,
              uint32_t * cb,
              uint32_t ng,
              uint32_t np,
              uint32_t nPts,
              uint32_t nDim,
              uint32_t nVec) {

  for (uint32_t s = 0; s < 2; s++ ) { // red-black sync

    CILK_FOR (uint32_t idual = 0; idual < (ng-3) ; idual += 6) { // coarse-grid

      for (uint32_t ifine = 0 ; ifine < 3 ; ifine++ ) { // fine-grid

        // get index of current grid box
        uint32_t i = 3*s+idual + ifine;

        // if above boundaries, break
        if (i > ng-4) break;

        // loop through all points inside box
        for (uint32_t k = 0; k < cb[i]; k++) {

          uint32_t f1, f2, f3;
          double d;
          double v1[4], v2[4], v3[4];
          
          f1 = (uint32_t) floor( y(0,ib[i]+k) );
          d = y(0,ib[i]+k)  - (double) f1;

          v1[0] = g2(1+d);
          v1[1] = g1(  d);
          v1[2] = g1(1-d);
          v1[3] = g2(2-d);

          f2 = (uint32_t) floor( y(1,ib[i]+k) );
          d = y(1,ib[i]+k)  - (double) f2;

          v2[0] = g2(1+d);
          v2[1] = g1(  d);
          v2[2] = g1(1-d);
          v2[3] = g2(2-d);

          f3 = (uint32_t) floor( y(2,ib[i]+k) );
          d = y(2,ib[i]+k)  - (double) f3;
      
          v3[0] = g2(1+d);
          v3[1] = g1(  d);
          v3[2] = g1(1-d);
          v3[3] = g2(2-d);

          for (uint32_t j = 0; j<nVec; j++){

            for (uint32_t idx3 = 0; idx3<4; idx3++){

              for (uint32_t idx2 = 0; idx2<4; idx2++){

                double qv2v3 = q(j,ib[i]+k) * v2[idx2] * v3[idx3];
        
                for (uint32_t idx1 = 0; idx1<4; idx1++){
                  V3(f1+idx1, f2+idx2, f3+idx3, j, 0) += qv2v3 * v1[idx1];
          
                } // (idx1)

              } // (idx2)

            } // (idx3)
            
          } // (j)
          
        } // (k)
        
      } // (ifine)
      
    } // (idual)
    
  } // (s)

}


void g2s1d( double * Phi,
            double * V,
            double * y,
            uint32_t ng,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  CILK_FOR (uint32_t i = 0; i<nPts; i++){

    uint32_t f1;
    double d;

    double v1[4];
      
    f1 = (uint32_t) floor( y(0,i) );
    d = y(0,i) - (double) f1;

    v1[0] = g2(1+d);
    v1[1] = g1(  d);
    v1[2] = g1(1-d);
    v1[3] = g2(2-d);

    for (uint32_t j = 0; j<nVec; j++){

      double accum = 0;

      for (uint32_t idx1 = 0; idx1<4; idx1++){
        accum += 
          V1(f1+idx1, j, 0) * v1[idx1];
            
      }

      Phi(j,i) = accum;

    }

  } // (i)

}


void g2s2d( double * Phi,
            double * V,
            double * y,
            uint32_t ng,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  CILK_FOR (uint32_t i = 0; i<nPts; i++){

    uint32_t f1, f2;
    double d;

    double v1[4];
    double v2[4];
      
    f1 = (uint32_t) floor( y(0,i) );
    d = y(0,i) - (double) f1;

    v1[0] = g2(1+d);
    v1[1] = g1(  d);
    v1[2] = g1(1-d);
    v1[3] = g2(2-d);

    f2 = (uint32_t) floor( y(1,i) );
    d = y(1,i) - (double) f2;
      
    v2[0] = g2(1+d);
    v2[1] = g1(  d);
    v2[2] = g1(1-d);
    v2[3] = g2(2-d);

    for (uint32_t j = 0; j<nVec; j++){

      double accum = 0;
      
      for (uint32_t idx2 = 0; idx2<4; idx2++){
        double qv2 = v2[idx2];
          
        for (uint32_t idx1 = 0; idx1<4; idx1++){
          accum += 
            V2(f1+idx1, f2+idx2, j, 0) * qv2 * v1[idx1];
            
        }

      }

      Phi(j,i) = accum;
      
    }

  } // (i)

}



void g2s3d( double * Phi,
            double * V,
            double * y,
            uint32_t ng,
            uint32_t nPts,
            uint32_t nDim,
            uint32_t nVec) {


  CILK_FOR (uint32_t i = 0; i<nPts; i++){

    uint32_t f1, f2, f3;
    double d;

    double v1[4];
    double v2[4];
    double v3[4];
      
    f1 = (uint32_t) floor( y(0,i) );
    d = y(0,i) - (double) f1;

    v1[0] = g2(1+d);
    v1[1] = g1(  d);
    v1[2] = g1(1-d);
    v1[3] = g2(2-d);

    f2 = (uint32_t) floor( y(1,i) );
    d = y(1,i) - (double) f2;
      
    v2[0] = g2(1+d);
    v2[1] = g1(  d);
    v2[2] = g1(1-d);
    v2[3] = g2(2-d);

    f3 = (uint32_t) floor( y(2,i) );
    d = y(2,i) - (double) f3;
      
    v3[0] = g2(1+d);
    v3[1] = g1(  d);
    v3[2] = g1(1-d);
    v3[3] = g2(2-d);
    
    for (uint32_t j = 0; j<nVec; j++){

      double accum = 0;

      for (uint32_t idx3 = 0; idx3<4; idx3++){
        
        for (uint32_t idx2 = 0; idx2<4; idx2++){

          double qv2v3 = v2[idx2] * v3[idx3];
          
          for (uint32_t idx1 = 0; idx1<4; idx1++){
            accum += 
            V3(f1+idx1, f2+idx2, f3+idx3, j, 0) * qv2v3 * v1[idx1];
            
          }

        }
        
      }

      Phi(j,i) = accum;

    }
        
  } // (i)

}

