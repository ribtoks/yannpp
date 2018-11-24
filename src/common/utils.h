#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

namespace yannpp {
    struct shape3d_t;

    namespace utils {
        int get_top_padding(shape3d_t const &input_shape,
                            shape3d_t const &filter_shape,
                            size_t stride_length);

        int get_left_padding(shape3d_t const &input_shape,
                             shape3d_t const &filter_shape,
                             size_t stride_length);
    }
}

#endif // UTILS_H
