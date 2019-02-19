#include <memory>
#include <string>
#include <initializer_list>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/layers/crossentropyoutputlayer.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/layers/poolinglayer.h>
#include <yannpp/network/activator.h>
#include <yannpp/network/network2.h>
#include <yannpp/optimizer/sdg_optimizer.h>

#include "parsing/mnist_dataset.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

static yannpp::activator_t<float> sigmoid_activator(yannpp::sigmoid_v<float>, yannpp::sigmoid_derivative_v<float>);
// derivative returns 1 because it is cancelled out when using cross-entropy
static yannpp::activator_t<float> softmax_activator(yannpp::stable_softmax_v<float>,
                                     [](yannpp::array3d_t<float> const &x){
    return yannpp::array3d_t<float>(yannpp::shape_row(x.size()), 1.0);});
static yannpp::activator_t<float> relu_activator(yannpp::relu_v<float>, yannpp::relu_v<float>);

using training_data_t = std::vector<std::tuple<yannpp::array3d_t<float>, yannpp::array3d_t<float>>>;

class MnistTests: public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        yannpp::mnist_dataset_t mnist_dataset(STRINGIZE(DATADIR));
        s_training_data = mnist_dataset.training_data();
    }

    static void TearDownTestSuite() {
        s_training_data.clear();
    }

protected:
    static training_data_t s_training_data;
};

training_data_t MnistTests::s_training_data;

TEST_F (MnistTests, LearnMnistDenseTest) {
    using namespace yannpp;

    size_t mini_batch_size = 10;
    float learning_rate = 0.01f;
    float decay_rate = 20.f;

    // reduce size for testing
    decltype (s_training_data) training_data(s_training_data.begin(),
                                             s_training_data.begin() + s_training_data.size()/10);

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

    size_t epochs = 2;

    network.init_layers();
    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);

    std::vector<size_t> eval_indices(training_data.size() / 6);
    // generate indices from 1 to the number of inputs
    std::iota(eval_indices.begin(), eval_indices.end(), 5*training_data.size() / 6);
    auto result = network.evaluate(training_data, eval_indices);

    ASSERT_GT(result, 800);
}

std::vector<yannpp::network2_t<float>::layer_type> create_dl_layers() {
    using namespace yannpp;
    std::vector<network2_t<float>::layer_type> layers = std::initializer_list<network2_t<float>::layer_type>(
    {
                        std::make_shared<convolution_layer_loop_t<float>>(
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
                        std::make_shared<crossentropy_output_layer_t<float>>()});
    return layers;
}

static yannpp::array3d_t<float> filter_initializer(yannpp::shape3d_t(5, 5, 1), 0.f, 1.f/5.f);
static yannpp::array3d_t<float> bias_initializer(yannpp::shape3d_t(1, 1, 1), 0.f);
static yannpp::array3d_t<float> fc1_initializer(yannpp::shape3d_t(30, 10*12*12, 1), 0.f, 1.f/sqrt(10*12));
static yannpp::array3d_t<float> fc2_initializer(yannpp::shape3d_t(10, 30, 1), 0.f, 1.f/sqrt(10));

void init_layers(std::vector<yannpp::network2_t<float>::layer_type> &layers) {
    using namespace yannpp;

    std::vector<array3d_t<float>> filters, biases;
    for (int i = 0; i < 10; i++) {
        filters.emplace_back(filter_initializer.clone());
        biases.emplace_back(bias_initializer.clone());
    }
    layers[0]->load(std::move(filters), std::move(biases));
    // layers[1]-> skip pooling layer
    layers[2]->load({fc1_initializer.clone()}, {array3d_t<float>(shape3d_t(30, 1, 1), 0.f)});
    layers[3]->load({fc2_initializer.clone()}, {array3d_t<float>(shape3d_t(10, 1, 1), 0.f)});
    // layers[4]-> skip crossentropy layer
}

TEST_F (MnistTests, DeepLearningLoopMnistTest) {
    using namespace yannpp;

    size_t mini_batch_size = 40;
    float learning_rate = 0.001;
    float decay_rate = 10.0;

    // reduce size for testing
    decltype (s_training_data) training_data(s_training_data.begin(),
                                             s_training_data.begin() + s_training_data.size()/30);

    sdg_optimizer_t<float> sdg_optimizer(mini_batch_size,
                                         training_data.size(),
                                         decay_rate,
                                         learning_rate);

    auto layers = create_dl_layers();
    init_layers(layers);
    network2_t<float> network(std::move(layers));

    size_t epochs = 2;

    network.init_layers();
    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);

    const size_t training_size = 5 * training_data.size() / 6;
    std::vector<size_t> eval_indices(training_data.size() - training_size);
    // generate indices from 1 to the number of inputs
    std::iota(eval_indices.begin(), eval_indices.end(), training_size);
    auto result = network.evaluate(training_data, eval_indices);

    ASSERT_GT(result, 2*eval_indices.size()/3);
}

TEST_F (MnistTests, DeepLearning2DMnistTest) {
    using namespace yannpp;

    size_t mini_batch_size = 40;
    float learning_rate = 0.001;
    float decay_rate = 10.0;

    // reduce size for testing
    decltype (s_training_data) training_data(s_training_data.begin(),
                                             s_training_data.begin() + s_training_data.size()/30);

    sdg_optimizer_t<float> sdg_optimizer(mini_batch_size,
                                         training_data.size(),
                                         decay_rate,
                                         learning_rate);

    auto layers = create_dl_layers();
    // replace loop convolution to 2D
    layers[0] = std::make_shared<convolution_layer_2d_t<float>>(
                                                                   shape3d_t(28, 28, 1), // input size
                                                                   shape3d_t(5, 5, 1), // filter size
                                                                   10, // filters count
                                                                   1, // stride length
                                                                   padding_type::valid,
                                                                   relu_activator);
    init_layers(layers);
    network2_t<float> network(std::move(layers));

    size_t epochs = 2;

    network.init_layers();
    network.train(training_data,
                  sdg_optimizer,
                  epochs,
                  mini_batch_size);

    const size_t training_size = 5 * training_data.size() / 6;
    std::vector<size_t> eval_indices(training_data.size() - training_size);
    // generate indices from 1 to the number of inputs
    std::iota(eval_indices.begin(), eval_indices.end(), training_size);
    auto result = network.evaluate(training_data, eval_indices);

    ASSERT_GT(result, 2*eval_indices.size()/3);
}
