#ifndef MAX_REDUCER_HPP
#define MAX_REDUCER_HPP


#include <limits>
#include <algorithm>

template<typename T> void min_limit(void *view) {
    *static_cast<T *>(view) = std::numeric_limits<T>::lowest();
}

template<typename T> void max_operation(void *left, void *right) {
    *static_cast<T *>(left) = std::max(*static_cast<T *>(left),*static_cast<T *>(right));
}

template<typename T> using max_reducer = T cilk_reducer(min_limit<T>, max_operation<T>);

#endif // MAX_REDUCER_HPP
