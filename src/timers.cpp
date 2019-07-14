#include "timers.hpp"

struct timeval tsne_start_timer(){
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv;
}

double tsne_stop_timer(const char * event_name, timeval begin){
  struct timeval end;
  gettimeofday(&end, nullptr);
  double stime = ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
    ((double) (end.tv_usec - begin.tv_usec) / 1000 );
  stime = stime / 1000;
#ifdef PRINT_DEBUG_TIME
  printf("%-20s : %8.4lf s\n",event_name, stime);
#endif
  return(stime);
}
