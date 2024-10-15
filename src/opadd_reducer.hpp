#ifndef OPADD_REDUCER_H
#define OPADD_REDUCER_H

#ifdef OPENCILK

template <typename T> static void zero(void *v) {
    *static_cast<T *>(v) = static_cast<T>(0);
}

template <typename T> static void plus(void *l, void *r) {
    *static_cast<T *>(l) += *static_cast<T *>(r);
}

template <typename T> using opadd_reducer = T cilk_reducer(zero<T>, plus<T>);

#else

template <typename T> using opadd_reducer = T;

#endif // OPENCILK

#endif // OPADD_REDUCER_H
