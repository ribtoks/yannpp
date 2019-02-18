#include <memory>
#include <string>
#include <initializer_list>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/network/activator.h>
#include <yannpp/layers/crossentropyoutputlayer.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/layers/poolinglayer.h>
#include <yannpp/network/network2.h>
#include <yannpp/optimizer/sdg_optimizer.h>

#include "parsing/mnist_dataset.h"

using namespace yannpp;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw std::runtime_error("Data root not specified through the command line");
    }

    std::string data_root(argv[1]);
    using namespace yannpp;
    mnist_dataset_t mnist_dataset(data_root);

    size_t mini_batch_size = 40;
    float learning_rate = 0.001;
    float decay_rate = 10.0;

    auto training_data = mnist_dataset.training_data();
    // reduce size for testing
    training_data.resize(training_data.size()/10);

    activator_t<float> sigmoid_activator(sigmoid_v<float>, sigmoid_derivative_v<float>);
    // derivative returns 1 because it is cancelled out when using cross-entropy
    activator_t<float> softmax_activator(stable_softmax_v<float>,
                                          [](array3d_t<float> const &x){
        return array3d_t<float>(shape_row(x.size()), 1.0);});
    activator_t<float> relu_activator(relu_v<float>, relu_v<float>);
    sdg_optimizer_t<float> sdg_optimizer(mini_batch_size,
                                        training_data.size(),
                                        decay_rate,
                                        learning_rate);
    network2_t<float> network(
                std::initializer_list<network2_t<float>::layer_type>(
    {
                        std::make_shared<convolution_layer_2d_t<float>>(
                        shape3d_t(28, 28, 1), // input size
                        shape3d_t(5, 5, 1), // filter size
                        10, // filters count
                        1, // stride length
                        padding_type::valid,
                        relu_activator),
                        std::make_shared<pooling_layer_t<float>>(
                        2, // window_size
                        2), // stride length
                        /*std::make_shared<convolution_layer_loop_t<float>>(
                        shape3d_t(12, 12, 20), // input size
                        shape3d_t(5, 5, 20), // filter size
                        20,
                        1, // stride length
                        padding_type::valid,
                        relu_activator),
                        std::make_shared<pooling_layer_t<float>>(
                        2, // window_size
                        2), // stride length*/
                        std::make_shared<fully_connected_layer_t<float>>(10*12*12, 30, relu_activator),
                        std::make_shared<fully_connected_layer_t<float>>(30, 10, softmax_activator),
                        std::make_shared<crossentropy_output_layer_t<float>>()}));

    size_t epochs = 60;

    network.init_layers();
    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);

    return 0;
}
