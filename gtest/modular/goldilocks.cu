#ifdef GOLDILOCKS_64

TEST(XpuModularGoldilocks, modOneIsZero) {
    
    ASSERT_EQ( 0 ,0);
    
}

TEST(XpuModularGoldilocks, singleModAdd) {
    /**
     * input should be 0...q-1
     >>> ((1<<16)+(1<<16)-2) % (1<<16)
    65534
    */
    using namespace goldilocks64_cpu; //TODO: change them to _gpu
    
    Data a = (1 << 16) - 1 , b = (1 << 16) - 1;
    Modulus q(1 << 16);
    GoldilocksOperations gred;
    std::cout <<"gred.add(a, b, q) " << gred.add(a, b, q) << std::endl;
    ASSERT_EQ( gred.add(a, b, q) , 65534);
    
}

TEST(XpuModularGoldilocks, multiplemodAddTest) {
    /**
     * input should be 0...q-1
     >>> [(x, ((1<<x)+(1<<x)-2) % (1<<x)) for x in range(10,58,2)]
     >>> for t in [(x, ((1<<x)+(1<<x)-2) % (1<<x)) for x in range(10,58,2)]:
            print("tupleVector.push_back(std::make_tuple({}, {}ULL));".format(t[0],t[1]))
     */
    using namespace goldilocks64_cpu; //TODO: change them to _gpu
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
    tupleVector.push_back(std::make_tuple(32, 4294967294ULL)); //fails here
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
        GoldilocksOperations gred;
        ASSERT_EQ( gred.add(a, b, q) , expected);

    }
    
}

#endif