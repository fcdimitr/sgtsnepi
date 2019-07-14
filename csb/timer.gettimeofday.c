#ifndef _MYCLOCK_
#define _MYCLOCK_


#include <sys/time.h>

struct timeval timer_ApplicationStartTime;
int timer_initialized = 0;

void timer_init(){
  if(timer_initialized){fprintf(stderr,"timer_init() must be called once and only once\n");exit(0);}
  timer_initialized = 1;
  struct timezone timer_TimeZone;
  gettimeofday(&timer_ApplicationStartTime,&timer_TimeZone);
}

double timer_seconds_since_init(){
  if(!timer_initialized){fprintf(stderr,"timer_init() must be called first\n");exit(0);}
  struct timezone timer_TimeZone;
  struct timeval  timer_CurrentTime;
  gettimeofday(&timer_CurrentTime,&timer_TimeZone);
  double rv = 1.0*(timer_CurrentTime.tv_sec-timer_ApplicationStartTime.tv_sec)+1e-6*(timer_CurrentTime.tv_usec-timer_ApplicationStartTime.tv_usec);
  return(rv);
}

#endif
