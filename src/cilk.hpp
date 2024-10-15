#ifndef CILK_HPP
#define CILK_HPP

#ifdef OPENCILK

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define CILK_FOR cilk_for
#define CILK_SPAWN cilk_spawn
#define CILK_SYNC cilk_sync

#else

#define CILK_FOR for
#define CILK_SPAWN
#define CILK_SYNC

#endif // OPENCILK

#endif // CILK_HPP
