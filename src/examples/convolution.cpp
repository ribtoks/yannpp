#include <memory>
#include <string>

#include "common/array3d.h"
#include "common/array3d_math.h"
#include "common/log.h"
#include "common/utils.h"

using namespace yannpp;

int main(int argc, char* argv[]) {
  std::vector<float> vx(3*3*3);
  for (int i = 0; i < vx.size(); i++) { vx[i] = (float)i; }

  std::vector<float> vw(5*5*3);
  for (int i = 0; i < vw.size(); i++) { vw[i] = (float)i; }

  array3d_t<float> filter_(shape3d_t(3, 3, 3), vx);
  array3d_t<float> input_(shape3d_t(5, 5, 3), vw);
    
  shape3d_t output_shape(input_.shape().x(), input_.shape().y(), 1);
  array3d_t<float> result(output_shape, 0.f);

  const int pad_x = utils::get_left_padding(input_.shape(), filter_.shape(), 1);
  const int pad_y = utils::get_top_padding(input_.shape(), filter_.shape(), 1);

  auto filter = filter_.slice();
  // 2D loop over the input and calculation convolution of input and current filter
  // convolution is S(i, j) = (I ∗ K)(i, j) = Sum[ I(m, n)K(i − m, j − n) ]
  // which is commutative i.e. (I ∗ K)(i, j) = Sum[ I(i - m, j - n)K(m, n) ]
  // where I is input and K is kernel (filter weights)
  for (int y = 0; y < output_shape.y(); y++) {
    int ys = y * 1 - pad_y;

    for (int x = 0; x < output_shape.x(); x++) {
      int xs = x * 1 - pad_x;
      // in this case cross-correlation (I(m, n)K(i + m, j + n)) is used
      // (kernel is not rot180() flipped for the convolution, not commutative)
      // previous formula (w*x + b) is used with convolution instead of product
      result(x, y, 0) =
          dot<float>(
            input_.slice(
              index3d_t(xs, ys, 0),
              index3d_t(xs + filter_.shape().x() - 1,
                        ys + filter_.shape().y() - 1,
                        input_.shape().z() - 1)),
            filter);
    }
  }

  log(result);

  return 0;
}
