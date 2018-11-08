# yannpp
This is an educational effort to help understand how deep neural networks work.

In order to achieve this goal I prepared a small number of selected educational materials and heavily documented pure C++ implementation of CNN that classifies MNIST digits.

# Understand
In order to fully understand what is going on, I would recommend doing following:

- read [great Michael Nielsen's online book](http://neuralnetworksanddeeplearning.com/) to understand all the basics
- read ["Backpropagation In Convolutional Neural Networks"](https://github.com/ribtoks/yannpp/blob/master/docs/Backpropagation%20In%20Convolutional%20Neural%20Networks%20-%20DeepGrid.pdf) pdf in the `docs/` to understand how to prove backpropagation equations for convolutional layers
- read ["A guide to convolution arithmetic"](https://github.com/ribtoks/yannpp/blob/master/docs/1603.07285.pdf) pdf in `docs/` to understand what is padding and how to convolve input and filter

After this you will be able to understand code in the repo.

# Get in
C++ code in the repo is simple enough to work in Windows/Mac/Linux. You can use CMake to compile it.

[![Build status](https://ci.appveyor.com/api/projects/status/p4coum570w8g3cxx/branch/master?svg=true)](https://ci.appveyor.com/project/Ribtoks/yannpp/branch/master) [![Build Status](https://travis-ci.org/ribtoks/yannpp.svg?branch=master)](https://travis-ci.org/ribtoks/yannpp)
![license](https://img.shields.io/badge/license-MIT-blue.svg) ![copyright](https://img.shields.io/badge/%C2%A9-Taras_Kushnir-blue.svg) ![language](https://img.shields.io/badge/language-c++-blue.svg) ![c++](https://img.shields.io/badge/std-c++11-blue.svg) 


# See
Main learning loop (as defined in `network2_t::backpropagate()`) looks like this:

    // feedforward input
    for (size_t i = 0; i < layers_size; i++) {
        input = layers_[i]->feedforward(input);
    }

    // backpropagate error
    array3d_t<network2_t::data_type> error(result);
    for (size_t i = layers_size; i-- > 0;) {
        error = layers_[i]->backpropagate(error);
    }

Because of this simplicity most interesting things are located in `src/layers/` directory that contains implementations of those `feedforward()` and `backpropagate()` methods for each layer. This codebase contains it's own greatly simplified `ndarray` as in Numpy and it's called `array3d_t`. Most useful feature of the array is the ability to slice parts of it's data as subarrays.

# Do
Feel free to experiment. For example, `examples/mnist_deeplearning.cpp` file specifically contains lots of experimental code (e.g. reducing size of the input to be able to experiement with network topology, commented layers in the network itself etc.) that can show you how to experiment. Experimentation is required to select hyperparameters, to see if your network converges etc.

# Get out
Feel free to say thank you it was useful. Also this code (as any other) may contain bugs or other problems - all contributions are highly welcome.

- [Fork](https://help.github.com/forking/) yannpp repository on GitHub
- Clone your fork locally
- Configure the upstream repo (`git remote add upstream git@github.com:ribtoks/yannpp.git`)
- Create local branch (`git checkout -b your_feature`)
- Work on your feature
- Push the branch to GitHub (`git push origin your_feature`)
- Send a [pull request](https://help.github.com/articles/using-pull-requests) on GitHub
