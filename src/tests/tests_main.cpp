#include <cstdlib>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    //::testing::GTEST_FLAG(catch_exceptions) = false;
    //::testing::GTEST_FLAG(filter) = "ConvolutionTests.*";
    ::testing::GTEST_FLAG(random_seed) = 123;
    srand(123);

    ::testing::InitGoogleTest(&argc, argv);
    srand(123);
    return RUN_ALL_TESTS();
}
