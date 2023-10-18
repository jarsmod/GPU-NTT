#include <gtest/gtest.h>
#include "ntt.cuh"


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

#ifdef BARRETT_64 //ajaveed todo: get rid of ifdef logic altogether

TEST(GpuModularBarret, modOneIsZero) {
    using namespace barrett64_gpu;
    
    Data a = 0, b = 0;
    Modulus c(1);
    BarrettOperations bred;
    
    ASSERT_EQ( bred.add(a, b, c) ,0);
    
}

TEST(GpuModularBarret, modFactorIsZero) {
    /**
     * input should be 0...q-1
     >>> ((1<<16)+(1<<16)-2) % (1<<16)
    65534
    */
    using namespace barrett64_gpu;
    
    Data a = (1 << 16) - 1 , b = (1 << 16) - 1;
    Modulus q(1 << 16);
    BarrettOperations bred;
    std::cout <<"bred.add(a, b, q) " << bred.add(a, b, q) << std::endl;
    ASSERT_EQ( bred.add(a, b, q) , 65534);
    
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}