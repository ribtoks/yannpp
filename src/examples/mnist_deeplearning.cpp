#include <memory>
#include <string>
#include <initializer_list>

#include "common/array3d.h"
#include "common/array3d_math.h"
#include "network/activator.h"
#include "layers/crossentropyoutputlayer.h"
#include "layers/fullyconnectedlayer.h"
#include "layers/convolutionlayer.h"
#include "layers/poolinglayer.h"
#include "network/network2.h"
#include "optimizer/sdg_optimizer.h"
#include "parsing/mnist_dataset.h"

using namespace yannpp;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw std::runtime_error("Data root not specified through the command line");
    }
    
    std::string data_root(argv[1]);
    mnist_dataset_t mnist_dataset(data_root);

    size_t mini_batch_size = 40;
    double learning_rate = 0.001;
    double decay_rate = 10.0;

    auto training_data = mnist_dataset.training_data();
    // reduce size for testing
    training_data.resize(training_data.size()/10);

    activator_t<double> sigmoid_activator(sigmoid_v<double>, sigmoid_derivative_v<double>);
    // derivative returns 1 because it is cancelled out when using cross-entropy
    activator_t<double> softmax_activator(stable_softmax_v<double>,
                                          [](array3d_t<double> const &x){
        return array3d_t<double>(shape_row(x.size()), 1.0);});
    activator_t<double> relu_activator(relu_v<double>, relu_v<double>);
    sdg_optimizer_t<double> sdg_optimizer(mini_batch_size,
                                        training_data.size(),
                                        decay_rate,
                                        learning_rate);
    network2_t<double> network(
                std::initializer_list<network2_t<double>::layer_type>(
    {
                        std::make_shared<convolution_layer_t<double>>(
                        shape3d_t(28, 28, 1), // input size
                        shape3d_t(5, 5, 1), // filter size
                        10, // filters count
                        1, // stride length
                        padding_type::valid,
                        relu_activator),
                        std::make_shared<pooling_layer_t<double>>(
                        2, // window_size
                        2), // stride length
                        /*std::make_shared<convolution_layer_t<double>>(
                        shape3d_t(12, 12, 20), // input size
                        shape3d_t(5, 5, 20), // filter size
                        20,
                        1, // stride length
                        padding_type::valid,
                        relu_activator),
                        std::make_shared<pooling_layer_t<double>>(
                        2, // window_size
                        2), // stride length*/
                        std::make_shared<fully_connected_layer_t<double>>(10*12*12, 30, relu_activator),
                        std::make_shared<fully_connected_layer_t<double>>(30, 10, softmax_activator),
                        std::make_shared<crossentropy_output_layer_t<double>>()}));

    size_t epochs = 60;

    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);
    
    return 0;
}
