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

TEST(GpuModularBarret, multiplemodAddTest) {
    /**
     * input should be 0...q-1
     >>> [(x, ((1<<x)+(1<<x)-2) % (1<<x)) for x in range(10,58,2)]
     >>> for t in [(x, ((1<<x)+(1<<x)-2) % (1<<x)) for x in range(10,58,2)]:
            print("tupleVector.push_back(std::make_tuple({}, {}ULL));".format(t[0],t[1]))
     */
    using namespace barrett64_gpu;
    std::vector<std::tuple<int, uint64_t>> tupleVector;

    // Populate the vector with tuples of int and uint64_t values
    tupleVector.push_back(std::make_tuple(10, 1022ULL));
    tupleVector.push_back(std::make_tuple(12, 4094ULL));
    tupleVector.push_back(std::make_tuple(14, 16382ULL));
    tupleVector.push_back(std::make_tuple(16, 65534ULL));
    tupleVector.push_back(std::make_tuple(18, 262142ULL));
    tupleVector.push_back(std::make_tuple(20, 1048574ULL));
    tupleVector.push_back(std::make_tuple(10, 1022ULL));
    tupleVector.push_back(std::make_tuple(12, 4094ULL));
    tupleVector.push_back(std::make_tuple(14, 16382ULL));
    tupleVector.push_back(std::make_tuple(16, 65534ULL));
    tupleVector.push_back(std::make_tuple(18, 262142ULL));
    tupleVector.push_back(std::make_tuple(20, 1048574ULL));
    tupleVector.push_back(std::make_tuple(22, 4194302ULL));
    tupleVector.push_back(std::make_tuple(24, 16777214ULL));
    tupleVector.push_back(std::make_tuple(26, 67108862ULL));
    tupleVector.push_back(std::make_tuple(28, 268435454ULL));
    tupleVector.push_back(std::make_tuple(30, 1073741822ULL));
    tupleVector.push_back(std::make_tuple(32, 4294967294ULL));
    tupleVector.push_back(std::make_tuple(34, 17179869182ULL));
    tupleVector.push_back(std::make_tuple(36, 68719476734ULL));
    tupleVector.push_back(std::make_tuple(38, 274877906942ULL));
    tupleVector.push_back(std::make_tuple(40, 1099511627774ULL));
    tupleVector.push_back(std::make_tuple(42, 4398046511102ULL));
    tupleVector.push_back(std::make_tuple(44, 17592186044414ULL));
    tupleVector.push_back(std::make_tuple(46, 70368744177662ULL));
    tupleVector.push_back(std::make_tuple(48, 281474976710654ULL));
    tupleVector.push_back(std::make_tuple(50, 1125899906842622ULL));
    tupleVector.push_back(std::make_tuple(52, 4503599627370494ULL));
    tupleVector.push_back(std::make_tuple(54, 18014398509481982ULL));
    tupleVector.push_back(std::make_tuple(56, 72057594037927934ULL));

    for (size_t i = 0; i < tupleVector.size(); ++i) {
        int bits;
        uint64_t expected;
        std::tie(bits, expected) = tupleVector[i];
        
        Data a = (1 << bits) - 1 , b = (1 << bits) - 1;
        Modulus q(1 << bits);
        BarrettOperations bred;
        ASSERT_EQ( bred.add(a, b, q) , expected);

    }
    
}

TEST(GpuModularBarret, multiplemodSubTest) {
    /**
     * input should be 0...q-1
     >>> for t in [ (x,((1<<x)-(1<<x-2)-2) % (1<<x)) for x in range(10,58,2)]:
...     print("tupleVector.push_back(std::make_tuple({}, {}ULL));".format(t[0],t[1]))
     */
    using namespace barrett64_gpu;
    std::vector<std::tuple<int, uint64_t>> tupleVector;

    // Populate the vector with tuples of int and uint64_t values
    tupleVector.push_back(std::make_tuple(10, 766ULL));
    tupleVector.push_back(std::make_tuple(12, 3070ULL));
    tupleVector.push_back(std::make_tuple(14, 12286ULL));
    tupleVector.push_back(std::make_tuple(16, 49150ULL));
    tupleVector.push_back(std::make_tuple(18, 196606ULL));
    tupleVector.push_back(std::make_tuple(20, 786430ULL));
    tupleVector.push_back(std::make_tuple(22, 3145726ULL));
    tupleVector.push_back(std::make_tuple(24, 12582910ULL));
    tupleVector.push_back(std::make_tuple(26, 50331646ULL));
    tupleVector.push_back(std::make_tuple(28, 201326590ULL));
    tupleVector.push_back(std::make_tuple(30, 805306366ULL));
    tupleVector.push_back(std::make_tuple(32, 3221225470ULL));
    tupleVector.push_back(std::make_tuple(34, 12884901886ULL));
    tupleVector.push_back(std::make_tuple(36, 51539607550ULL));
    tupleVector.push_back(std::make_tuple(38, 206158430206ULL));
    tupleVector.push_back(std::make_tuple(40, 824633720830ULL));
    tupleVector.push_back(std::make_tuple(42, 3298534883326ULL));
    tupleVector.push_back(std::make_tuple(44, 13194139533310ULL));
    tupleVector.push_back(std::make_tuple(46, 52776558133246ULL));
    tupleVector.push_back(std::make_tuple(48, 211106232532990ULL));
    tupleVector.push_back(std::make_tuple(50, 844424930131966ULL));
    tupleVector.push_back(std::make_tuple(52, 3377699720527870ULL));
    tupleVector.push_back(std::make_tuple(54, 13510798882111486ULL));
    tupleVector.push_back(std::make_tuple(56, 54043195528445950ULL));

    for (size_t i = 0; i < tupleVector.size(); ++i) {
        int bits;
        uint64_t expected;
        std::tie(bits, expected) = tupleVector[i];
        
        Data a = (1 << bits) - 1 , b = (1 << (bits-2)) - 1;
        Modulus q(1 << bits);
        BarrettOperations bred;
        ASSERT_EQ( bred.sub(a, b, q) , expected);

    }
    
}

TEST(GpuModularBarret, multiplemodMultTest) {
    /**
     * input should be 0...q-1
     >>> for t in [ (x,(((1<<x)-1) * ((1<<x-2)-1)) % (1<<x)) for x in range(10,58,2)]:
...     print("tupleVector.push_back(std::make_tuple({}, {}ULL));".format(t[0],t[1]))
     */
    using namespace barrett64_gpu;
    std::vector<std::tuple<int, uint64_t>> tupleVector;

    // Populate the vector with tuples of int and uint64_t values
    tupleVector.push_back(std::make_tuple(10, 769ULL));
    tupleVector.push_back(std::make_tuple(12, 3073ULL));
    tupleVector.push_back(std::make_tuple(14, 12289ULL));
    tupleVector.push_back(std::make_tuple(16, 49153ULL));
    tupleVector.push_back(std::make_tuple(18, 196609ULL));
    tupleVector.push_back(std::make_tuple(20, 786433ULL));
    tupleVector.push_back(std::make_tuple(22, 3145729ULL));
    tupleVector.push_back(std::make_tuple(24, 12582913ULL));
    tupleVector.push_back(std::make_tuple(26, 50331649ULL));
    tupleVector.push_back(std::make_tuple(28, 201326593ULL));
    tupleVector.push_back(std::make_tuple(30, 805306369ULL));
    tupleVector.push_back(std::make_tuple(32, 3221225473ULL));
    tupleVector.push_back(std::make_tuple(34, 12884901889ULL));
    tupleVector.push_back(std::make_tuple(36, 51539607553ULL));
    tupleVector.push_back(std::make_tuple(38, 206158430209ULL));
    tupleVector.push_back(std::make_tuple(40, 824633720833ULL));
    tupleVector.push_back(std::make_tuple(42, 3298534883329ULL));
    tupleVector.push_back(std::make_tuple(44, 13194139533313ULL));
    tupleVector.push_back(std::make_tuple(46, 52776558133249ULL));
    tupleVector.push_back(std::make_tuple(48, 211106232532993ULL));
    tupleVector.push_back(std::make_tuple(50, 844424930131969ULL));
    tupleVector.push_back(std::make_tuple(52, 3377699720527873ULL));
    tupleVector.push_back(std::make_tuple(54, 13510798882111489ULL));
    tupleVector.push_back(std::make_tuple(56, 54043195528445953ULL));

    for (size_t i = 0; i < tupleVector.size(); ++i) {
        int bits;
        uint64_t expected;
        std::tie(bits, expected) = tupleVector[i];
        
        Data a = (1 << bits) - 1 , b = (1 << (bits-2)) - 1;
        Modulus q(1 << bits);
        BarrettOperations bred;
        ASSERT_EQ( bred.sub(a, b, q) , expected);

    }
    
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}