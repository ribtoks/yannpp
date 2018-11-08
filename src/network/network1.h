#ifndef NETWORK1_H
#define NETWORK1_H

#include <initializer_list>
#include <vector>
#include <tuple>
#include "common/array3d.h"

class network1_t {
public:
    using t_d = array3d_t<double>;
    using training_data = std::vector<std::tuple<t_d, t_d>>;

public:
    network1_t(std::initializer_list<size_t> layers);

public:
    // train network using stochastic gradient descent
    // number of epochs, minibatch size, learning rate
    // weight decay are hyperparameters
    void train_sgd(const training_data &data,
                   size_t epochs,
                   size_t mini_batch_size,
                   double eta,
                   double lambda);

private:
    // evaluates number of correctly classified inputs (validation data)
    size_t evaluate(const training_data &data, const std::vector<size_t> &indices) const;

    // feeds input a to the network and returns output
    t_d feedforward(t_d a) const;

    // updates network weights and biases using one
    // iteration of gradient descent using mini_batch of inputs and outputs
    void update_mini_batch(const training_data &data,
                           const std::vector<size_t> &indices,
                           double eta,
                           double lambda);

    // runs a loop of propagation of inputs and backpropagation of errors
    // back to the beginning with weights and biases updates as a result
    void backpropagate(t_d const &input,
                       t_d const &result,
                       std::vector<t_d> &nabla_b,
                       std::vector<t_d> &nabla_w);

    t_d &activate(t_d &z) const;
    t_d &derivative(t_d &z) const;
    t_d cost_derivative(const t_d &actual, const t_d &expected) const;

private:
    //  dimensions of layers
    std::vector<size_t> layers_;
    // bias(i) is a vector of biases of neurons in layer (i)
    std::vector<t_d> biases_;
    // weight(i) is a matrix of weights between layer (i) and (i + 1)
    std::vector<t_d> weights_;
};

#endif // NETWORK1_H
