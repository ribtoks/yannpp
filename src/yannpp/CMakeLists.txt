cmake_minimum_required(VERSION 3.6)

project(yannpp CXX)

set (CMAKE_CXX_STANDARD 11)

set(SOURCES
    common/cpphelpers.h
    common/cpphelpers.cpp
    common/shape.h
    common/array3d.h
    common/array3d_math.h
    common/log.h
    common/log.cpp
    common/utils.h
    common/utils.cpp
    optimizer/sdg_optimizer.h
    optimizer/optimizer.h
    network/network2.h
#    network/network1.h
#    network/network1.cpp
    layers/fullyconnectedlayer.h
    layers/poolinglayer.h
    layers/crossentropyoutputlayer.h
    layers/convolutionlayer.h
    layers/layer_base.h
    network/activator.h)

add_library(yannpp SHARED STATIC ${SOURCES})

target_include_directories(yannpp PRIVATE ${YANNPP_SOURCE_DIR})

install(TARGETS yannpp DESTINATION lib)
