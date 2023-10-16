#include <gtest/gtest.h>
#include "../src/ntt.cuh"


TEST(DeviceTests, MaxGridSizeProp0is2147483647) {
    CudaDevice();
    int device = 0; // Assuming you are using device 0
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int expectedResult = 2147483647;
    ASSERT_EQ(prop.maxGridSize[0], expectedResult);

    //std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;

}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}