#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/log.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/optimizer/optimizer.h>

static yannpp::activator_t<float> relu_activator(yannpp::relu_v<float>, yannpp::relu_v<float>);

bool arrays_equal(yannpp::array3d_t<float> const &a, yannpp::array3d_t<float> const &b) {
    bool equal = false;

    do {
        if (a.shape() != b.shape()) { break; }

        auto &adata = a.data();
        auto &bdata = b.data();
        if (adata.size() != bdata.size()) {
            yannpp::log("Arrays shapes are different");
            break;
        }

        float eps = 0.0000001f;
        bool anyFailure = false;
        const size_t size = adata.size();
        for (size_t i = 0; i < size; i++) {
            if (fabs(adata[i] - bdata[i]) > eps) {
                yannpp::log("Difference at %d: %.6f != %.6f", i, adata[i], bdata[i]);
                anyFailure = true;
                break;
            }
        }

        equal = !anyFailure;
    } while (false);

    return equal;
}

int fill_array(yannpp::array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        slice.at(*it) = i;
    }

    return (i - 1);
}

yannpp::array3d_t<float> create_input(yannpp::shape3d_t const &shape) {
    return yannpp::array3d_t<float>(shape, 0.f, 1.f);
}

yannpp::array3d_t<float> create_error(yannpp::shape3d_t const &shape) {
    return yannpp::array3d_t<float>(shape, 0.5f, 0.8f);
}

std::vector<yannpp::array3d_t<float>> create_filters(int count, yannpp::shape3d_t const &shape) {
    std::vector<yannpp::array3d_t<float>> filters;

    for (auto fi = 0; fi < count; fi++) {
        filters.emplace_back(shape, 0.f);
    }

    int index = 0;

    for (auto height = 0; height < shape.y(); height++) {
        for (auto width = 0; width < shape.x(); width++) {
            for (auto depth = 0; depth < shape.z(); depth++) {
                for (auto fi = 0; fi < count; fi++, index++) {
                    filters[fi](height, width, depth) = index;
                }
            }
        }
    }

    return filters;
}

std::vector<yannpp::array3d_t<float>> create_biases(int count) {
    std::vector<yannpp::array3d_t<float>> biases;

    for (auto fi = 0; fi < count; fi++) {
        biases.emplace_back(yannpp::shape3d_t(1, 1, 1), fi + 0.5f);
    }

    return biases;
}

// this class exists in order not to create friend class
// to access inner nabla_w and nabla_b
class fake_optimizer_t: public yannpp::optimizer_t<float> {
private:
    using nabla_array = std::vector<yannpp::array3d_t<float>>;
public:
    virtual void update_bias(yannpp::array3d_t<float> &, yannpp::array3d_t<float> &nabla_b) const override {
        const_cast<nabla_array&>(this->nabla_b_).push_back(nabla_b);
    }

    virtual void update_weights(yannpp::array3d_t<float> &, yannpp::array3d_t<float> &nabla_w) const override {
        const_cast<nabla_array&>(this->nabla_w_).push_back(nabla_w);
    }

public:
    nabla_array const &get_nabla_w() const { return nabla_w_; }
    nabla_array const &get_nabla_b() const { return nabla_b_; }

private:
    nabla_array nabla_w_;
    nabla_array nabla_b_;
};

TEST (ConvolutionTests, LoopFeedForwardWith2DSamePaddingTest) {
    using namespace yannpp;

    shape3d_t filter_shape(3, 3, 5);
    shape3d_t input_shape(5, 5, 5);
    int filters_number = 10;
    int stride_length = 1;

    convolution_layer_loop_t<float> loop(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::same,
                relu_activator);
    loop.load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));

    convolution_layer_2d_t<float> matrix(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::same,
                relu_activator);
    matrix.load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));

    ASSERT_TRUE(loop.get_output_shape() == matrix.get_output_shape());

    auto input = create_input(input_shape);
    ASSERT_TRUE(arrays_equal(loop.feedforward(input),
                             matrix.feedforward(input)));
}

TEST (ConvolutionTests, LoopFeedForwardWith2DValidPaddingTest) {
    using namespace yannpp;

    shape3d_t filter_shape(3, 3, 5);
    shape3d_t input_shape(5, 5, 5);
    int filters_number = 10;
    int stride_length = 1;

    convolution_layer_loop_t<float> loop(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::valid,
                relu_activator);
    loop.load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));

    convolution_layer_2d_t<float> matrix(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::valid,
                relu_activator);
    matrix.load(create_filters(filters_number, filter_shape),
                create_biases(filters_number));

    ASSERT_TRUE(loop.get_output_shape() == matrix.get_output_shape());

    auto input = create_input(input_shape);
    ASSERT_TRUE(arrays_equal(loop.feedforward(input),
                             matrix.feedforward(input)));
}

using conv_loop_ptr = std::shared_ptr<yannpp::convolution_layer_loop_t<float>>;
using conv_matrix_ptr = std::shared_ptr<yannpp::convolution_layer_2d_t<float>>;

void prepare_layers_for_comparison(conv_loop_ptr &loop,
                                   conv_matrix_ptr &matrix) {
    using namespace yannpp;

    shape3d_t filter_shape(3, 3, 5);
    shape3d_t input_shape(5, 5, 5);
    int filters_number = 10;
    int stride_length = 1;
    auto input = create_input(input_shape);

    loop = std::make_shared<convolution_layer_loop_t<float>>(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::same,
                relu_activator);
    loop->load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));
    auto error = create_error(loop->get_output_shape());
    loop->init();
    loop->feedforward(input);
    loop->backpropagate(error);

    matrix = std::make_shared<convolution_layer_2d_t<float>>(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::same,
                relu_activator);
    matrix->load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));
    matrix->init();
    matrix->feedforward(input);
    matrix->backpropagate(error);
}

TEST (ConvolutionTests, NablaWeightBackpropagateWithSamePaddingTest) {
    using namespace yannpp;

    conv_loop_ptr loop;
    conv_matrix_ptr matrix;
    prepare_layers_for_comparison(loop, matrix);

    fake_optimizer_t matrix_optimizer;
    matrix->optimize(matrix_optimizer);

    fake_optimizer_t loop_optimizer;
    loop->optimize(loop_optimizer);

    auto &loop_nabla_w = loop_optimizer.get_nabla_w();
    auto &matrix_nabla_w = matrix_optimizer.get_nabla_w();

    ASSERT_EQ(loop_nabla_w.size(), matrix_nabla_w.size());
    ASSERT_GT(loop_nabla_w.size(), 0);

    for (size_t i = 0; i < loop_nabla_w.size(); i++) {
        ASSERT_TRUE(arrays_equal(loop_nabla_w[i], matrix_nabla_w[i])) << "Arrays are not equal at " << i;
    }
}

TEST (ConvolutionTests, NablaBiasBackpropagateWithSamePaddingTest) {
    using namespace yannpp;

    conv_loop_ptr loop;
    conv_matrix_ptr matrix;
    prepare_layers_for_comparison(loop, matrix);

    fake_optimizer_t matrix_optimizer;
    matrix->optimize(matrix_optimizer);

    fake_optimizer_t loop_optimizer;
    loop->optimize(loop_optimizer);

    auto &loop_nabla_b = loop_optimizer.get_nabla_b();
    auto &matrix_nabla_b = matrix_optimizer.get_nabla_b();

    ASSERT_EQ(loop_nabla_b.size(), matrix_nabla_b.size());
    ASSERT_GT(loop_nabla_b.size(), 0);

    for (size_t i = 0; i < loop_nabla_b.size(); i++) {
        ASSERT_TRUE(arrays_equal(loop_nabla_b[i], matrix_nabla_b[i])) << "Arrays are not equal at " << i;
    }
}

TEST (ConvolutionTests, LoopBackpropagateWith2DSamePaddingTest) {
    using namespace yannpp;

    shape3d_t filter_shape(3, 3, 5);
    shape3d_t input_shape(5, 5, 5);
    int filters_number = 10;
    int stride_length = 1;

    auto input = create_input(input_shape);

    convolution_layer_loop_t<float> loop(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::valid,
                relu_activator);
    loop.load(create_filters(filters_number, filter_shape),
              create_biases(filters_number));
    loop.init();
    loop.feedforward(input);

    convolution_layer_2d_t<float> matrix(
                input_shape,
                filter_shape,
                filters_number,
                stride_length,
                padding_type::valid,
                relu_activator);
    matrix.load(create_filters(filters_number, filter_shape),
                create_biases(filters_number));
    matrix.init();
    matrix.feedforward(input);

    ASSERT_TRUE(loop.get_output_shape() == matrix.get_output_shape());

    auto error = create_error(loop.get_output_shape());
    ASSERT_TRUE(arrays_equal(loop.backpropagate(error),
                             matrix.backpropagate(error)));
}
