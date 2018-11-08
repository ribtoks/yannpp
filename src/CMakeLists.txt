cmake_minimum_required(VERSION 3.6)

project(mnist_training CXX)

set (CMAKE_CXX_STANDARD 11)

set(SOURCES
    parsing/bmp_image.h
    parsing/bmp_image.cpp
    parsing/parsed_images.h
    parsing/parsed_images.cpp
    parsing/parsed_labels.h
    parsing/parsed_labels.cpp
    common/cpphelpers.h
    common/cpphelpers.cpp
    common/shape.h
    common/array3d.h
    common/array3d_math.h
    common/array3d_math.cpp
    common/log.h
    common/log.cpp
    common/utils.h
    common/utils.cpp
    optimizer/sdg_optimizer.h
    optimizer/optimizer.h
    network/network2.h
    network/network2.cpp
#    network/network1.h
#    network/network1.cpp
    layers/fullyconnectedlayer.h
    layers/poolinglayer.h
    layers/crossentropyoutputlayer.h
    layers/convolutionlayer.h
    layers/layer_base.h
    network/activator.h
    parsing/mnist_dataset.h
    parsing/mnist_dataset.cpp)

#add_executable(mnist_training ${SOURCES} examples/mnist_simple.cpp)
#add_executable(mnist_training ${SOURCES} examples/mnist_fc_network.cpp)
add_executable(mnist_training ${SOURCES} examples/mnist_deeplearning.cpp)

target_include_directories(mnist_training PRIVATE .)
