#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <vector>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/shape.h>
#include <yannpp/common/utils.h>
#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>
#include <yannpp/network/activator.h>
#include <yannpp/optimizer/optimizer.h>

namespace yannpp {
#define FILTER_DIM(input, filter, stride) (((input) - (filter))/(stride) + 1)

    enum struct padding_type {
        valid, // only positions where the kernel lies entirely within the image
        same  // output is equal in size to the input
    };

    template<typename T>
    class convolution_layer_base_t: public layer_base_t<T> {
    public:
        convolution_layer_base_t(shape3d_t const &input_shape,
                                 shape3d_t const &filter_shape,
                                 int filters_number,
                                 int stride_length,
                                 padding_type padding,
                                 activator_t<T> const &activator,
                                 layer_metadata_t const &metadata={}):
            layer_base_t<T>(metadata),
            input_shape_(input_shape),
            filter_shape_(filter_shape),
            // shape of the result of convolution of input and kernel/filter
            conv_shape_(FILTER_DIM(input_shape.x(), filter_shape.x(), stride_length),
                        FILTER_DIM(input_shape.y(), filter_shape.y(), stride_length),
                        filters_number),
            stride_(stride_length, stride_length, 0),
            padding_(padding),
            activator_(activator)
        {
            assert(filter_shape.z() == input_shape.z());
        }

    public:
        virtual void init() override {
            assert(filter_weights_.size() == filter_biases_.size());

            const int filters_number = conv_shape_.z();

            if (filter_weights_.empty()) {
                filter_weights_.reserve(filters_number);
                filter_biases_.reserve(filters_number);

                // all neurons in each filter share same weights and bias
                for (int i = 0; i < filters_number; i++) {
                    filter_weights_.emplace_back(
                                filter_shape_,
                                T(0), T(1)/sqrt((T)filter_shape_.capacity()));
                    filter_biases_.emplace_back(shape_row(1), 0);
                }
            }

            if (nabla_weights_.empty()) {
                nabla_weights_.reserve(filters_number);
                nabla_biases_.reserve(filters_number);

                for (int i = 0; i < filters_number; i++) {
                    nabla_weights_.emplace_back(filter_shape_, 0);
                    nabla_biases_.emplace_back(shape_row(1), 0);
                }
            }
        }

        virtual void optimize(optimizer_t<T> const &strategy) override {
            const size_t size = filter_weights_.size();
            for (size_t i = 0; i < size; i++) {
                strategy.update_weights(filter_weights_[i], nabla_weights_[i]);
                strategy.update_bias(filter_biases_[i], nabla_biases_[i]);
                nabla_weights_[i].reset(0);
                nabla_biases_[i].reset(0);
            }
        }

        virtual void load(std::vector<array3d_t<T>> &&weights, std::vector<array3d_t<T>> &&biases) override {
            assert(weights.size() == /*filters_number*/ conv_shape_.z());
            assert(biases.size() == /*filters_number*/ conv_shape_.z());
            assert(std::all_of(weights.begin(), weights.end(), [this](array3d_t<T> const &f) {
                       return (f.shape() == this->filter_shape_);
                   }));
            assert(std::all_of(biases.begin(), biases.end(), [this](array3d_t<T> const &b) {
                       return (b.size() == 1 && b.shape().dim() == 0);
                   }));

            filter_weights_ = std::move(weights);
            filter_biases_ = std::move(biases);
        }

    protected:
        shape3d_t get_output_shape() const {
            if (padding_ == padding_type::valid) { return conv_shape_; }

            double width = ceil(double(input_shape_.x()) / double(stride_.x()));
            double height = ceil(double(input_shape_.y()) / double(stride_.y()));
            return shape3d_t((int)width, (int)height, filter_weights_.size());
        }

        int get_top_padding() const {
            if (padding_ == padding_type::valid) { return 0; }
            return utils::get_top_padding(input_shape_, filter_shape_, stride_.y());
        }

        int get_left_padding() const {
            if (padding_ == padding_type::valid) { return 0; }
            return utils::get_left_padding(input_shape_, filter_shape_, stride_.x());
        }

    protected:
        shape3d_t input_shape_;
        shape3d_t filter_shape_;
        // shape which is the result of the convolution of image and filter
        shape3d_t conv_shape_;
        point3d_t<int> stride_;
        const padding_type padding_;
        activator_t<T> const &activator_;
        std::vector<array3d_t<T>> filter_weights_;
        std::vector<array3d_t<T>> filter_biases_;
        // calculation support
        array3d_t<T> input_, output_;
        std::vector<array3d_t<T>> nabla_weights_;
        std::vector<array3d_t<T>> nabla_biases_;
    };

    template<typename T>
    class convolution_layer_loop_t: public convolution_layer_base_t<T> {
    public:
        // use same constructor
        using convolution_layer_base_t<T>::convolution_layer_base_t;

    public:
        virtual array3d_t<T> feedforward(array3d_t<T> const &input) override {
            assert(input.shape() == this->input_shape_);

            this->input_ = input.clone();
            const shape3d_t output_shape = this->get_output_shape();
            array3d_t<T> result(output_shape, 0);

            const int pad_x = this->get_left_padding();
            const int pad_y = this->get_top_padding();

            const int fsize = this->filter_weights_.size();
            auto &filter_shape = this->filter_shape_;
            auto &input_shape = this->input_.shape();
            // perform convolution for each filter
            for (int fi = 0; fi < fsize; fi++) {
                auto filter = this->filter_weights_[fi].slice();
                auto &bias = this->filter_biases_[fi](0);
                // 2D loop over the input and calculation convolution of input and current filter
                // convolution is S(i, j) = (I ∗ K)(i, j) = Sum[ I(m, n)K(i − m, j − n) ]
                // which is commutative i.e. (I ∗ K)(i, j) = Sum[ I(i - m, j - n)K(m, n) ]
                // where I is input and K is kernel (filter weights)
                for (int y = 0; y < output_shape.y(); y++) {
                    int ys = y * this->stride_.y() - pad_y;

                    for (int x = 0; x < output_shape.x(); x++) {
                        int xs = x * this->stride_.x() - pad_x;
                        // in this case cross-correlation (I(m, n)K(i + m, j + n)) is used
                        // (kernel is not rot180() flipped for the convolution, not commutative)
                        // previous formula (w*x + b) is used with convolution instead of product
                        result(x, y, fi) =
                                bias +
                                dot<T>(
                                    this->input_.slice(
                                        index3d_t(xs, ys, 0),
                                        index3d_t(xs + filter_shape.x() - 1,
                                                  ys + filter_shape.y() - 1,
                                                  input_shape.z() - 1)),
                                    filter);
                    }
                }
            }

            this->output_ = std::move(result);
            return this->activator_.activate(this->output_);
        }

        virtual array3d_t<T> backpropagate(array3d_t<T> const &error) override {
            // error shape was already transformed in the prev layer as delta(l+1)(*)rot180(w(l+1))
            assert(error.shape() == this->output_.shape());
            auto &error_shape = error.shape();
            // gradietns with regards to input of this layer
            array3d_t<T> delta;
            delta = this->activator_.derivative(this->output_); delta.element_mul(error);

            const int pad_x = this->get_left_padding();
            const int pad_y = this->get_top_padding();

            const size_t fsize = this->filter_weights_.size();
            auto &filter_shape = this->filter_shape_, &input_shape = this->input_shape_;
            auto &stride = this->stride_;
            // calculate nabla_w for each filter
            for (int fi = 0; fi < fsize; fi++) {
                auto &nabla_w = this->nabla_weights_[fi];
                auto &nabla_b = this->nabla_biases_[fi](0);
                auto delta_fi = delta.slice(dim_type::Z, fi, fi);

                for (int z = 0; z < input_shape.z(); z++) {
                    // convolution of input and filter gives us output (same as error size)
                    // and convolution of input and error gives us filter size
                    for (int y = 0; y < filter_shape.y(); y++) {
                        int ys = y * stride.y() - pad_y;

                        for (int x = 0; x < filter_shape.x(); x++) {
                            int xs = x * stride.x() - pad_x;

                            // dC/db = delta(l)
                            nabla_b += delta(x, y, fi);
                            // dC/dw = a(l-1) (x) delta(l)
                            nabla_w(x, y, z) +=
                                    dot<T>(
                                        this->input_.slice(
                                            index3d_t(xs, ys, z),
                                            index3d_t(xs + error_shape.x() - 1,
                                                      ys + error_shape.y() - 1,
                                                      z)),
                                        delta_fi);
                        }
                    }
                }
            }

            array3d_t<T> delta_next(this->input_shape_, T(0));

            // use 'full' convolution (http://www.johnloomis.org/ece563/notes/filter/conv/convolution.html)
            // so we need to set appropriate padding
            const int weight_pad_x = utils::get_left_padding(error_shape, filter_shape, stride.x());
            const int weight_pad_y = utils::get_top_padding(error_shape, filter_shape, stride.y());

            // input gradient of next layer is scaled by weights gradient of this layer
            // gradient for the next layer is delta(l) (*) rot180(w(l))
            // so for delta we apply "full" convolution with filter
            for (size_t fi = 0; fi < fsize; fi++) {
                for (int z = 0; z < input_shape.z(); z++) {
                    auto weights = this->filter_weights_[fi].slice(dim_type::Z, z, z);

                    // result of the convolution of delta and weights will be input size
                    for (int y = 0; y < input_shape.y(); y++) {
                        int ys = y*stride.y() - weight_pad_y;

                        for (int x = 0; x < input_shape.x(); x++) {
                            int xs = x*stride.x() - weight_pad_x;

                            delta_next(x, y, z) =
                                    dot<T>(
                                        delta.slice(
                                            index3d_t(xs, ys, fi),
                                            index3d_t(xs + filter_shape.x() - 1,
                                                      ys + filter_shape.y() - 1,
                                                      fi)),
                                        weights);
                        }
                    }
                }
            }

            return delta_next;
        }
    };

    template<typename T>
    class convolution_layer_2d_t: public convolution_layer_base_t<T> {
    public:
        // use same constructor
        using convolution_layer_base_t<T>::convolution_layer_base_t;

    public:
        virtual array3d_t<T> feedforward(array3d_t<T> const &input) override {
            assert(input.shape() == this->input_shape_);
            this->input_ = input.clone();
            /*
             * Flattens the filter to a 2-D matrix with shape
             *   [filter_height * filter_width * in_channels, output_channels].
             * Extracts image patches from the input to form a
             *   [batch, out_height, out_width, filter_height * filter_width * in_channels].
             * For each patch, right-multiplies the filter matrix and the image patch vector.
             */
            auto patches = image_patches();
            auto filters = flat_filters();

            const shape3d_t output_shape = this->get_output_shape();
            std::vector<T> result;
            result.reserve(output_shape.capacity());

            const size_t psize = patches.size();
            for (size_t i = 0; i < psize; i++) {
                auto conv = dot21(filters, patches[i]);
                result.insert(result.end(), conv.data().begin(), conv.data().end());
            }

            this->output_ = array3d_t<T>(output_shape, std::move(result));
            return this->activator_.activate(this->output_);
        }

        virtual array3d_t<T> backpropagate(array3d_t<T> const &error) override {
            // TODO: implement this
            return error;
        }

    private:
        array3d_t<T> flat_filters() {
            const int fsize = this->filter_weights_.size();
            const int flength = this->filter_shape_.capacity();
            std::vector<T> filters_matrix;
            filters_matrix.reserve(fsize * flength);
            for (int fi = 0; fi < fsize; fi++) {
                auto &data = this->filter_weights_[fi].data();
                filters_matrix.insert(filters_matrix.end(), data.begin(), data.end());
            }
            return array3d_t<T>(shape3d_t(fsize, flength, 1), std::move(filters_matrix));
        }

        std::deque<array3d_t<T>> image_patches() {
            std::deque<array3d_t<T>> patches;
            const shape3d_t output_shape = this->get_output_shape();
            auto &filter_shape = this->filter_shape_;

            const int pad_x = this->get_left_padding();
            const int pad_y = this->get_top_padding();

            for (int x = 0; x < output_shape.x(); x++) {
                int xs = x * this->stride_.x() - pad_x;

                for (int y = 0; y < output_shape.y(); y++) {
                    int ys = y * this->stride_.y() - pad_y;

                    patches.emplace_back(
                                shape3d_t(filter_shape.capacity(), 1, 1),
                                std::move(
                                    this->input_.slice(
                                        index3d_t(xs, ys, 0),
                                        index3d_t(xs + filter_shape.x() - 1,
                                                  ys + filter_shape.y() - 1,
                                                  this->input_shape_.z() - 1))
                                    .extract()));
                }
            }

            return patches;
        }
    };
}

#endif // CONVOLUTIONLAYER_H
