#include "network1.h"

#include <numeric>

#include "common/array3d_math.h"
#include "common/cpphelpers.h"
#include "common/log.h"

std::vector<network1_t::t_d> copy_shapes(const std::vector<network1_t::t_d> &from) {
    std::vector<network1_t::t_d> to;
    to.reserve(from.size());

    for (auto &a: from) {
        to.emplace_back(a.shape(), 0);
    }

    return to;
}

network1_t::network1_t(std::initializer_list<size_t> layers):
    layers_(layers)
{
    const size_t layers_count = layers_.size();

    biases_.reserve(layers_count - 1);
    // first layer of neurons is an input
    // and doesn't have any biases and weights
    for (size_t i = 1; i < layers_count; i++) {
        size_t dim = layers_[i];
        biases_.emplace_back(shape_row(dim), 0.0, 1.0);
    }

    weights_.reserve(layers_count - 1);
    // weights_[i] means weights between
    // layer (i) and layer (i+1)
    // let w be this matrix weights_[i]
    // then w[j][k] means weight of connection
    // between k-th neuron in layer (i) and
    // j-th neuron in layer (i+1)
    for (size_t i = 0; i < layers_count - 1; i++) {
        size_t height = layers_[i+1];
        size_t width = layers_[i];
        weights_.emplace_back(shape_matrix(height, width),
                              0.0,  // mean
                              1.0/sqrt((double)width));  // std deviation
    }

    assert(weights_.size() == biases_.size());
    assert(weights_.size() == layers_.size() - 1);
}

void network1_t::train_sgd(const training_data &data,
                          size_t epochs,
                          size_t minibatch_size,
                          double eta,
                          double lambda) {
    // big chunk of data is used for training while
    // small chunk - for validation after some epochs
    const size_t training_size = 5 * data.size() / 6;
    std::vector<size_t> eval_indices(data.size() - training_size);
    std::iota(eval_indices.begin(), eval_indices.end(), training_size);

    for (size_t j = 0; j < epochs; j++) {
        auto indices_batches = batch_indices(training_size, minibatch_size);
        for (auto &indices: indices_batches) {
            update_mini_batch(data, indices, eta, lambda);
        }

        if (j % 2 == 0) {
            auto result = evaluate(data, eval_indices);
            log("Epoch %d: %d / %d", j, result, eval_indices.size());
        } else {
            log("Epoch %d ended", j);
        }
    }

    auto result = evaluate(data, eval_indices);
    log("End result: %d / %d", result, eval_indices.size());
}

size_t network1_t::evaluate(const training_data &data, const std::vector<size_t> &indices) const {
    size_t count = 0;
    for (auto i: indices) {
        network1_t::t_d result = feedforward(std::get<0>(data[i]));
        assert(result.size() == std::get<1>(data[i]).size());
        if (argmax1d(result) == argmax1d(std::get<1>(data[i]))) { count++; }
    }
    return count;
}

network1_t::t_d network1_t::feedforward(network1_t::t_d a) const {
    assert(weights_.size() == biases_.size());
    assert(a.size() == layers_[0]);
    const size_t size = weights_.size();
    for (size_t i = 0; i < size; i++) {
        // a = sigma(w*a + b)
        a = activate( dot_2d_1d(weights_[i], a).add(biases_[i]) );
    }
    return a;
}

void network1_t::update_mini_batch(const network1_t::training_data &data,
                                  const std::vector<size_t> &indices,
                                  double eta,
                                  double lambda) {
    auto nabla_b = copy_shapes(biases_);
    auto nabla_w = copy_shapes(weights_);

    auto delta_nabla_b = copy_shapes(biases_);
    auto delta_nabla_w = copy_shapes(weights_);

    const size_t size = layers_.size() - 1;
    for (auto i: indices) {
        auto &input = std::get<0>(data[i]);
        auto &result = std::get<1>(data[i]);

        for (auto &dnb: delta_nabla_b) { dnb.reset(0.0); }
        for (auto &dnw: delta_nabla_w) { dnw.reset(0.0); }

        backpropagate(input, result, delta_nabla_b, delta_nabla_w);

        // partial derivatives dC/dw and dC/db are averaged
        // across all training examples (div by minimatch_size later)
        for (size_t i = 0; i < size; i++) {
            nabla_b[i].add(delta_nabla_b[i]);
            nabla_w[i].add(delta_nabla_w[i]);
        }
    }

    double minibatch_size = indices.size() + 0.0;
    // eta is learning rate
    double scale = eta/minibatch_size;
    // lambda is decay rate hyperparameter
    double decay = 1.0 - eta*lambda/data.size();

    // general gradient descent formulas:
    // w = w - eta/minibatch_size * gradient_w
    // b = b - eta/minibatch_size * gradient_b
    for (size_t i = 0; i < size; i++) {
        biases_[i].add(nabla_b[i].mul(-scale));
        weights_[i].mul(decay).add(nabla_w[i].mul(-scale));
    }
}

void network1_t::backpropagate(const network1_t::t_d &input,
                              const network1_t::t_d &result,
                              std::vector<network1_t::t_d> &nabla_b,
                              std::vector<network1_t::t_d> &nabla_w) {
    std::vector<network1_t::t_d> zs { };
    std::vector<network1_t::t_d> activations { input };
    const size_t size = weights_.size();
    const size_t layers_count = layers_.size();

    // forward pass
    for (size_t i = 0; i < size; i++) {
        auto z = dot_2d_1d(weights_[i], activations.back()).add(biases_[i]);
        zs.push_back(z);
        activations.push_back(activate(z));
    }

    // delta(L) = cost_deriv [X] sigma_deriv(z(L))
    // cross-entropy
    auto delta = cost_derivative(activations.back(), result);

    // MSE:
    //auto delta = cost_derivative(activations.back(), result)
    //    .element_mul(activation_derivative(zs.back()));

    // delta(l) = (w(l+1) * delta(l+1)) [X] sigma_deriv(z(l))
    // dC/db = delta(l)
    // dC/dw = a(l-1) * delta(l)
    nabla_b[size - 1] = delta;
    nabla_w[size - 1] = dot_transpose(delta, activations[layers_count - 2]);

    // propagating errors backwards
    for (size_t i = 1; i < size; i++) {
        size_t l = size - 1 - i;
        delta = transpose_dot(weights_[l + 1], delta)
                .element_mul(activation_derivative(zs[l]));
        nabla_b[l] = delta;
        nabla_w[l] = dot_transpose(delta, activations[l]);
    }
}

network1_t::t_d &network1_t::activate(network1_t::t_d &z) const {
    return z.apply(sigmoid);
}

network1_t::t_d &network1_t::activation_derivative(network1_t::t_d &z) const {
    return z.apply(sigmoid_derivative);
}

network1_t::t_d network1_t::cost_derivative(const t_d &actual, const t_d &expected) const {
    network1_t::t_d result(actual);
    result.subtract(expected);
    return result;
}
