#ifndef ILAYER_H
#define ILAYER_H

#include "common/array3d.h"

template<typename T>
class optimizer_t;

template<typename T>
class layer_base_t {
public:
    virtual ~layer_base_t() {}
    // input is the output of the previous layer
    virtual array3d_t<T> feedforward(array3d_t<T> const &input) = 0;
    // error is the gradient with regards to input
    virtual array3d_t<T> backpropagate(array3d_t<T> const &error) = 0;
    virtual void update_weights(optimizer_t<T> const &) = 0;
};


#endif // ILAYER_H
