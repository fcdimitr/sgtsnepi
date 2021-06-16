/*!
  \file   timers.hpp
  \brief  Routines for timing and profiling functions.

  \author Dimitris Floros
  \date   2019-06-24
*/


#ifndef TIMERS_HPP
#define TIMERS_HPP

#include <stdio.h>
#include <sys/time.h>


struct timeval tsne_start_timer();

double tsne_stop_timer(const char * event_name, timeval begin);


#endif /* TIMERS_HPP */
