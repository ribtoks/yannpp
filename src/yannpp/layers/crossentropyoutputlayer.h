#ifndef CROSSENTROPYOUPTUTLAYER_H
#define CROSSENTROPYOUPTUTLAYER_H

#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>
#include <yannpp/network/activator.h>
#include <yannpp/optimizer/optimizer.h>

namespace yannpp {
    template <typename T>
    class crossentropy_output_layer_t : public layer_base_t<T> {
    public:
        crossentropy_output_layer_t(layer_metadata_t const &metadata={}):
            layer_base_t<T>(metadata)
        { }

    public:
        virtual void init() override { }

        virtual array3d_t<T> feedforward(array3d_t<T> &&input) override {
            last_activation_ = std::move(input);
            return last_activation_;
        }

        virtual array3d_t<T> backpropagate(array3d_t<T> &&result) override {
            // delta(L) = cost_deriv [X] activation_deriv(z(L))
            // cross-entropy derivative is [a(x) - y]
            last_activation_.subtract(result);
            return last_activation_;
        }

        virtual void optimize(optimizer_t<T> const &) override {}
        virtual void load(std::vector<array3d_t<T>> &&, std::vector<array3d_t<T>> &&) override {}

    private:
        array3d_t<T> last_activation_;
    };
}

#endif // CROSSENTROPYOUPTUTLAYER_H
