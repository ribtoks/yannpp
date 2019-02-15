#include <iostream>
#include <vector>

#include "common/array3d.h"
#include "common/array3d_math.h"
#include "common/shape.h"
#include "common/log.h"
#include "layers/fullyconnectedlayer.h"
#include "network/activator.h"

int fill_array(yannpp::array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        auto index = *it;
        printf("(%d, %d, %d) = %d\n", index.x(), index.y(), index.z(), i);
        slice.at(*it) = i;
    }

    return (i - 1);
}

int main() {
    using namespace yannpp;

    activator_t<float> relu_activator(relu_v<float>, relu_v<float>);
    fully_connected_layer_t<float> dense(3, 2, relu_activator);

    array3d_t<float> weight(shape3d_t(3, 2, 1), 0.f);
    fill_array(weight);
    log(weight);
    for (auto &v: weight.data()) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    array3d_t<float> bias(shape3d_t(2, 1, 1), 0.f);

    std::vector<array3d_t<float>> weights = { weight };
    std::vector<array3d_t<float>> biases = { bias };

    dense.load(weights, biases);

    array3d_t<float> input(shape3d_t(3, 1, 1), 1.f);
    auto output = dense.feedforward(input);
    log(output);

    log(weight.flatten());

    for (auto &v: weight.data()) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}
