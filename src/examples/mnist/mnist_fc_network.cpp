#include <memory>
#include <string>
#include <initializer_list>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/layers/crossentropyoutputlayer.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/network/activator.h>
#include <yannpp/network/network2.h>
#include <yannpp/optimizer/sdg_optimizer.h>

#include "parsing/mnist_dataset.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw std::runtime_error("Data root not specified through the command line");
    }

    std::string data_root(argv[1]);
    using namespace yannpp;
    mnist_dataset_t mnist_dataset(data_root);

    size_t mini_batch_size = 10;
    float learning_rate = 0.005f;
    float decay_rate = 20.f;

    auto training_data = mnist_dataset.training_data();
    activator_t<float> sigmoid_activator(sigmoid_v<float>, sigmoid_derivative_v<float>);
    // derivative returns 1 because it is cancelled out when using cross-entropy
    activator_t<float> softmax_activator(stable_softmax_v<float>,
                                          [](array3d_t<float> const &x){
        return array3d_t<float>(shape_row(x.size()), 1.0);});

    sdg_optimizer_t<float> sdg_optimizer(mini_batch_size,
                                         training_data.size(),
                                         decay_rate,
                                         learning_rate);
    network2_t<float> network(
                std::initializer_list<network2_t<float>::layer_type>(
    {
                        std::make_shared<fully_connected_layer_t<float>>(28*28, 30, sigmoid_activator),
                        std::make_shared<fully_connected_layer_t<float>>(30, 10, softmax_activator),
                        std::make_shared<crossentropy_output_layer_t<float>>()}));

    size_t epochs = 60;

    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);

    return 0;
}
