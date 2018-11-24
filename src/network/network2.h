#ifndef NETWORK2_H
#define NETWORK2_H

#include <initializer_list>
#include <vector>
#include <tuple>
#include <memory>

#include "optimizer/optimizer.h"
#include "layers/fullyconnectedlayer.h"
#include "network/activator.h"

namespace yannpp {
    class network2_t {
    public:
        using data_type = double;
        using t_d = array3d_t<data_type>;
        using training_data = std::vector<std::tuple<t_d, t_d>>;
        using layer_type = std::shared_ptr<layer_base_t<data_type>>;

    public:
        network2_t(std::initializer_list<layer_type> layers);

    public:
        void train(training_data const &data,
                   optimizer_t<data_type> const &strategy,
                   size_t epochs,
                   size_t minibatch_size);

    private:
        // feeds input a to the network and returns output
        t_d feedforward(t_d const &a);

        // evaluates number of correctly classified inputs (validation data)
        size_t evaluate(training_data const &data, std::vector<size_t> const &indices);

        // updates network weights and biases using one
        // iteration of gradient descent using mini_batch of inputs and outputs
        void update_mini_batch(training_data const &data,
                               std::vector<size_t> const &indices,
                               optimizer_t<network2_t::data_type> const &strategy);

        // runs a loop of propagation of inputs and backpropagation of errors
        // back to the beginning with weights and biases updates as a result
        void backpropagate(t_d const &x, t_d const &result);

    private:
        //  dimensions of layers
        std::vector<std::shared_ptr<layer_base_t<data_type>>> layers_;
    };
}

#endif // NETWORK2_H
