cmake_minimum_required(VERSION 3.6)

project(yannpp_tests_project CXX)

set(SOURCES
    ${MNIST_SOURCE_DIR}/parsing/mnist_dataset.h
    ${MNIST_SOURCE_DIR}/parsing/mnist_dataset.cpp
    ${MNIST_SOURCE_DIR}/parsing/parsed_labels.h
    ${MNIST_SOURCE_DIR}/parsing/parsed_labels.cpp
    ${MNIST_SOURCE_DIR}/parsing/parsed_images.h
    ${MNIST_SOURCE_DIR}/parsing/parsed_images.cpp
    tests_main.cpp
    tests_mnist.cpp)

add_executable(yannpp_test ${SOURCES})

# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
# endif()

FILE(TO_NATIVE_PATH ${DATA_DIR} DATA_ROOT)

if(WIN32)
  STRING(REGEX REPLACE "\\\\" "\\\\\\\\" DATA_ROOT ${DATA_ROOT} )
endif()

add_definitions(-DDATADIR=${DATA_ROOT})
message("Data root: ${DATA_ROOT}")

target_include_directories(yannpp_test PRIVATE ${YANNPP_SOURCE_DIR})
target_include_directories(yannpp_test PRIVATE ${MNIST_SOURCE_DIR})
target_include_directories(yannpp_test PRIVATE ${GTEST_SOURCE_DIR}/googletest/include)
target_include_directories(yannpp_test PRIVATE ${GTEST_SOURCE_DIR}/include)

# GTest needs threading support

target_link_libraries(yannpp_test gtest_main)
target_link_libraries(yannpp_test yannpp)

add_test(NAME YannppTest COMMAND yannpp_test)
