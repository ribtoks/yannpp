cmake_minimum_required(VERSION 3.1)

project(YANNPP)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD_REQUIRED on)

#set(GTEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendors/gtest)

if (MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
   set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif (MSVC)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

enable_testing()

add_subdirectory(src)
