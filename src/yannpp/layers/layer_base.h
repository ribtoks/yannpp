#ifndef ILAYER_H
#define ILAYER_H

#include <vector>

#include <yannpp/common/array3d.h>
#include <yannpp/layers/layer_metadata.h>

namespace yannpp {
    template<typename T>
    class optimizer_t;

    template<typename T>
    class layer_base_t {
    public:
        layer_base_t(layer_metadata_t const &m={}): metadata_(m) {}
        virtual ~layer_base_t() {}
        // input is the output of the previous layer
        virtual array3d_t<T> feedforward(array3d_t<T> const &input) = 0;
        // error is the gradient with regards to input
        virtual array3d_t<T> backpropagate(array3d_t<T> const &error) = 0;
        virtual void load(std::vector<array3d_t<T>> &&weights, std::vector<array3d_t<T>> &&biases) = 0;
        virtual void optimize(optimizer_t<T> const &) = 0;

    public:
        layer_metadata_t const &get_metadata() const { return metadata_; }

    private:
        layer_metadata_t metadata_;
    };
}

#endif // ILAYER_H
