#include <gtest/gtest.h>
#include "ntt.cuh"


#include "misc/device.cu"
#include "modular/barret.cu"
#include "modular/goldilocks.cu"
#include "modular/plantard.cu"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}