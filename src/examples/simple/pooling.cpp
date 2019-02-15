#include <memory>
#include <string>
#include <vector>

#include "common/array3d.h"
#include "common/array3d_math.h"
#include "layers/poolinglayer.h"
#include "common/log.h"
#include "common/utils.h"

using namespace yannpp;

int fill_array(array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        slice.at(*it) = (float)i;
    }

    return (i - 1);
}

array3d_t<float> maxpool2d(array3d_t<float> const &input, int window_size_, int stride) {
    point3d_t<int> stride_(stride, stride, 1);
    auto &input_shape_ = input.shape();
    // downsample input using window with step stride
    shape3d_t output_shape(POOL_DIM(input_shape_.x(), window_size_, stride_.x()),
                           POOL_DIM(input_shape_.y(), window_size_, stride_.y()),
                           input_shape_.z());
    array3d_t<float> result(output_shape, 0.f);
    array3d_t<index3d_t> max_index_(output_shape, index3d_t(0, 0, 0));

    // z axis corresponds to each filter from convolution layer
    for (int z = 0; z < output_shape.z(); z++) {
        // 2D loop over convoluted image from each filter
        for (int y = 0; y < output_shape.y(); y++) {
            int ys = y * stride_.y();

            for (int x = 0; x < output_shape.x(); x++) {
                int xs = x * stride_.x();
                // pooling layer does max-pooling, selecting a maximum
                // activation within the bounds of it's "window"
                auto input_slice = const_cast<array3d_t<float>&>(input)
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

array3d_t<float> create_input(shape3d_t const &shape) {
    array3d_t<float> arr(shape, 0.f);
    fill_array(arr);
    return arr;
}

int main(int argc, char* argv[]) {
    using namespace yannpp;

//    [[[[18. 19. 20.]
//       [21. 22. 23.]
//       [24. 25. 26.]
//       [27. 28. 29.]]

//      [[33. 34. 35.]
//       [36. 37. 38.]
//       [39. 40. 41.]
//       [42. 43. 44.]]

//      [[48. 49. 50.]
//       [51. 52. 53.]
//       [54. 55. 56.]
//       [57. 58. 59.]]

//      [[63. 64. 65.]
//       [66. 67. 68.]
//       [69. 70. 71.]
//       [72. 73. 74.]]]]
    log(maxpool2d(create_input(shape3d_t(5, 5, 10)),
                  2,
                  2));

    return 0;
}
