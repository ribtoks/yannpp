#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/log.h>
#include <yannpp/layers/convolutionlayer.h>

static yannpp::activator_t<float> relu_activator(yannpp::relu_v<float>, yannpp::relu_v<float>);

bool arrays_equal(yannpp::array3d_t<float> const &a, yannpp::array3d_t<float> const &b) {
    bool equal = false;

    do {
        if (a.shape() != b.shape()) { break; }

        auto &adata = a.data();
        auto &bdata = b.data();
        if (adata.size() != bdata.size()) { break; }

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
    yannpp::array3d_t<float> arr(shape, 0.f);
    fill_array(arr);
    return arr;
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
        biases.emplace_back(yannpp::shape3d_t(1, 1, 1), 0.f);
    }

    return biases;
}

TEST (ConvolutionTests, LoopFeedForwardSameWith2DSamePaddingTest) {
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

    ASSERT_TRUE(arrays_equal(loop.feedforward(input),
                             matrix.feedforward(input)));
}
