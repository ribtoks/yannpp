#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <functional>

#include <yannpp/optimizer/optimizer.h>
#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/log.h>
#include <yannpp/common/shape.h>
#include <yannpp/network/activator.h>
#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>

namespace yannpp {
    template<typename T = double>
    class fully_connected_layer_t: public layer_base_t<T> {
    public:
        fully_connected_layer_t(size_t layer_in,
                                size_t layer_out,
                                activator_t<T> const &activator,
                                layer_metadata_t const &metadata = {}):
            layer_base_t<T>(metadata),
            activator_(activator),
            input_shape_(layer_out, layer_in, 1)
        { }

    public:
        virtual void init() override {
            const int layer_in = input_shape_.y(), layer_out = input_shape_.x();

            if (weights_.size() == 0) {
                weights_ = array3d_t<T>(
                               shape3d_t(layer_out, layer_in, 1),
                               T(0), T(1)/sqrt((T)layer_in));
            }

            if (bias_.size() == 0) {
                bias_ = array3d_t<T>(
                            shape_row(layer_out),
                            T(0), T(1));
            }

            nabla_w_ = array3d_t<T>(shape3d_t(layer_out, layer_in, 1), 0);
            nabla_b_ = array3d_t<T>(shape_row(layer_out), 0);
        }

        virtual array3d_t<T> feedforward(array3d_t<T> const &input) override {
            input_shape_ = input.shape();
            input_ = input.clone();
            input_.flatten();
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

        virtual void optimize(optimizer_t<T> const &strategy) override {
            strategy.update_bias(bias_, nabla_b_);
            strategy.update_weights(weights_, nabla_w_);
            nabla_b_.reset(0);
            nabla_w_.reset(0);
        }

    public:
        virtual void load(std::vector<array3d_t<T>> &&weights, std::vector<array3d_t<T>> &&biases) override {
            const int layer_in = input_shape_.y(), layer_out = input_shape_.x();
            shape3d_t weight_shape(layer_out, layer_in, 1);

            assert(!weights.empty());
            assert(!biases.empty());
            assert(weights[0].shape() == weight_shape);
            assert(biases[0].shape() == shape_row(layer_out));

            weights_ = std::move(weights[0]);
            bias_ = std::move(biases[0]);
        }

    private:
        // own data
        array3d_t<T> weights_;
        array3d_t<T> bias_;
        activator_t<T> const &activator_;
        // calculation support
        shape3d_t input_shape_;
        array3d_t<T> output_, input_;
        array3d_t<T> nabla_w_;
        array3d_t<T> nabla_b_;
    };
}

#endif // FULLY_CONNECTED_LAYER_H
