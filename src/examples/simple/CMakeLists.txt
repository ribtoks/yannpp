cmake_minimum_required(VERSION 3.6)

project(simple_example CXX)

set (CMAKE_CXX_STANDARD 11)

add_executable(simple_example ${SOURCES} convolution.cpp)

target_include_directories(simple_example PRIVATE ${YANNPP_SOURCE_DIR})

target_link_libraries(simple_example yannpp)
