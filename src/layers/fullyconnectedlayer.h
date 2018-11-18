#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <functional>

#include "optimizer/optimizer.h"
#include "common/array3d.h"
#include "common/array3d_math.h"
#include "common/log.h"
#include "common/shape.h"
#include "network/activator.h"
#include "layers/layer_base.h"

template<typename T = double>
class fully_connected_layer_t: public layer_base_t<T> {
public:
    fully_connected_layer_t(size_t layer_in,
                            size_t layer_out,
                            activator_t<T> const &activator):
        dimension_(layer_out),
        weights_(
            shape_matrix(layer_out, layer_in),
            T(0), T(1)/sqrt((T)layer_in)),
        bias_(
            shape_row(layer_out),
            T(0), T(1)),
        output_(shape_row(layer_out), 0),
        nabla_w_(shape_matrix(layer_out, layer_in), 0),
        nabla_b_(shape_row(layer_out), 0),
        activator_(activator),
        input_shape_(layer_out, layer_in, 1)
    { }

public:
    virtual array3d_t<T> feedforward(array3d_t<T> const &input) override {
        input_shape_ = input.shape();
        input_ = input.flatten();
        // z = w*a + b
        output_ = dot21(weights_, input_); output_.add(bias_);
        return activator_.activate(output_);
    }

    virtual array3d_t<T> backpropagate(array3d_t<T> const &error) override {
        array3d_t<T> delta, delta_next, delta_nabla_w;
        // delta(l) = (w(l+1) * delta(l+1)) [X] derivative(z(l))
        // (w(l+1) * delta(l+1)) comes as the gradient (error) from the "previous" layer
        delta = activator_.derivative(output_); delta.element_mul(error);
        // dC/db = delta(l)
        nabla_b_.add(delta);
        // dC/dw = a(l-1) * delta(l)
        delta_nabla_w = outer_product(delta, input_);
        nabla_w_.add(delta_nabla_w);
        // w(l) * delta(l)
        delta_next = transpose_dot21(weights_, delta);
        delta_next.reshape(input_shape_);
        return delta_next;
    }

    virtual void update_weights(optimizer_t<T> const &strategy) override {
        strategy.update_bias(bias_, nabla_b_);
        strategy.update_weights(weights_, nabla_w_);
		nabla_b_.reset(0);
		nabla_w_.reset(0);
    }

public:
    void load(array3d_t<T> &&weights, array3d_t<T> &&bias) {
        assert(weights_.shape() == weights.shape());
        assert(bias_.shape() == bias.shape());
        weights_ = std::move(weights);
        bias_ = std::move(bias);
    }

private:
    // own data
    size_t dimension_;
    array3d_t<T> weights_;
    array3d_t<T> bias_;
    activator_t<T> const &activator_;
    // calculation support
    shape3d_t input_shape_;
    array3d_t<T> output_, input_;
    array3d_t<T> nabla_w_;
    array3d_t<T> nabla_b_;
};

#endif // FULLY_CONNECTED_LAYER_H
