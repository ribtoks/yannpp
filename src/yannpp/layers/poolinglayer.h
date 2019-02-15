#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include <yannpp/common/shape.h>
#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>

namespace yannpp {
#define POOL_DIM(input, pool, stride) (((input) - (pool))/(stride) + 1)

    template <typename T>
    class pooling_layer_t: public layer_base_t<T> {
    public:
        pooling_layer_t(size_t window_size,
                        int stride_length,
                        layer_metadata_t const &metadata = {}):
            layer_base_t<T>(metadata),
            window_size_(window_size),
            input_shape_(0, 0, 0),
            stride_(stride_length, stride_length, 0)
        { }

    public:
        virtual void init() override { }

        virtual array3d_t<T> feedforward(array3d_t<T> const &input) override {
            input_shape_ = input.shape();
            // downsample input using window with step stride
            shape3d_t output_shape(POOL_DIM(input_shape_.x(), window_size_, stride_.x()),
                                   POOL_DIM(input_shape_.y(), window_size_, stride_.y()),
                                   input_shape_.z());
            array3d_t<T> result(output_shape, T(0));
            max_index_ = array3d_t<index3d_t>(output_shape, index3d_t(0, 0, 0));

            // z axis corresponds to each filter from convolution layer
            for (int z = 0; z < output_shape.z(); z++) {
                // 2D loop over convoluted image from each filter
                for (int y = 0; y < output_shape.y(); y++) {
                    int ys = y * stride_.y();

                    for (int x = 0; x < output_shape.x(); x++) {
                        int xs = x * stride_.x();
                        // pooling layer does max-pooling, selecting a maximum
                        // activation within the bounds of it's "window"
                        auto input_slice = const_cast<array3d_t<T>&>(input)
                                           .slice(
                                               index3d_t(xs, ys, z),
                                               index3d_t(xs + window_size_ - 1,
                                                         ys + window_size_ - 1,
                                                         z));
                        max_index_(x, y, z) = input_slice.argmax();
                        result(x, y, z) = input_slice.at(max_index_(x, y, z));
                    }
                }
            }

            return result;
        }

        virtual array3d_t<T> backpropagate(array3d_t<T> const &error) override {
            auto &error_shape = error.shape();
            array3d_t<T> output(input_shape_, T(0));
            assert(error.shape() == max_index_.shape());

            // z axis corresponds to each filter from convolution layer
            for (int z = 0; z < error_shape.z(); z++) {
                // 2D loop same as in feedforward()
                for (int y = 0; y < error_shape.y(); y++) {
                    int ys = y * stride_.y();

                    for (int x = 0; x < error_shape.x(); x++) {
                        int xs = x * stride_.x();

                        // same slice as input used for max() calculation
                        output.slice(
                                    index3d_t(xs, ys, z),
                                    index3d_t(xs + window_size_ - 1,
                                              ys + window_size_ - 1,
                                              z))
                                .at(max_index_(x, y, z)) = error(x, y, z);
                    }
                }
            }

            return output;
        }

        virtual void optimize(optimizer_t<T> const &) override {
            // no weight update is done in pooling layer
        }
        virtual void load(std::vector<array3d_t<T>> &&, std::vector<array3d_t<T>> &&) override {}

    private:
        size_t window_size_;
        shape3d_t input_shape_;
        point3d_t<int> stride_;
        array3d_t<index3d_t> max_index_;
    };
}

#endif // POOLINGLAYER_H
