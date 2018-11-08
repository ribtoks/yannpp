#include "utils.h"
#include "shape.h"

namespace utils {
    int get_top_padding(shape3d_t const &input_shape,
                        shape3d_t const &filter_shape,
                        size_t stride_length) {
        int padding = 0;
        // use tensorflow approach https://www.tensorflow.org/api_guides/python/nn#Convolution
        if (input_shape.y() % stride_length == 0) {
            padding = filter_shape.y() - stride_length;
        } else {
            padding = filter_shape.y() - (input_shape.y() % stride_length);
        }

        if (padding < 0) { padding = 0; }

        // return uneven padding (more padding to the bottom)
        return padding / 2;
    }

    int get_left_padding(shape3d_t const &input_shape,
                         shape3d_t const &filter_shape,
                         size_t stride_length) {
        int padding = 0;
        // use tensorflow approach https://www.tensorflow.org/api_guides/python/nn#Convolution
        if (input_shape.x() % stride_length == 0) {
            padding = filter_shape.x() - stride_length;
        } else {
            padding = filter_shape.x() - (input_shape.x() % stride_length);
        }

        if (padding < 0) { padding = 0; }

        // return uneven padding (more padding to the right)
        return padding / 2;
    }
}
