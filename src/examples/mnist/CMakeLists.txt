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
    parsing/mnist_dataset.h
    parsing/mnist_dataset.cpp)

#add_executable(mnist_training ${SOURCES} mnist_simple.cpp)
#add_executable(mnist_training ${SOURCES} mnist_fc_network.cpp)
add_executable(mnist_training ${SOURCES} mnist_deeplearning.cpp)

target_include_directories(mnist_training PRIVATE ${YANNPP_SOURCE_DIR})
target_include_directories(mnist_training PRIVATE .)

target_link_libraries(mnist_training yannpp)
