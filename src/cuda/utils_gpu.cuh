
#ifndef UTILS_GPU_CUH
#define UTILS_GPU_CUH

#define FULL_WARP_MASK 0xFFFFFFFF
template <class T>
__device__ T warp_reduce(T val){
  for(int offset=32/2;offset>0;offset/=2){
    val+=__shfl_down_sync(FULL_WARP_MASK,val,offset);

  }
  return val;
}
#endif
