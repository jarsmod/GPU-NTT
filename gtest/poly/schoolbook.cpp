/*
>>> a = 313709975877732223
>>> b = 246731823781677491
>>> a*b
77402234486818923332408775844492493
>>> (a*b) % 576460752303415297
16423228685580265
*/
TEST(SchoolBookPolyXply, single) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
    
    // std::cout <<" +v+ " << parameters.modulus.value 
    //           <<" +b+ " << parameters.modulus.bit
    //           <<" +u+ " << parameters.modulus.mu
            //  << std::endl;
    
    printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    std::vector<Data> input1{
        313709975877732223ULL,
    };

    std::vector<Data> input2{
        246731823781677491ULL,
    };

    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
    
    // printf("schoolbook_result\n");
    // printVector(schoolbook_result);

    
    ASSERT_EQ( schoolbook_result[0] ,16423228685580265);
    
}


TEST(SchoolBookPolyXply, onepoly) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
    
    // printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    std::vector<Data> input1{
        313709975877732223ULL,
        243906629715325917ULL,
        533808423925998667ULL
    };

    std::vector<Data> input2{
        1ULL,
        0ULL,
        0ULL
    };

    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
    
    ASSERT_EQ( std::equal(input1.begin(), input1.end(), schoolbook_result.begin()), true );
    
}

/**
 >>> a
array([313709975877732223, 243906629715325917, 533808423925998667],
      dtype=uint64)
>>> (a * 313709975877732223) % 576460752303415297
array([431893979263004911, 250850038660605336, 117502165775093281],
      dtype=uint64)

*/

TEST(SchoolBookPolyXply, second) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        313709975877732223ULL,
        243906629715325917ULL,
        533808423925998667ULL
    };

    std::vector<Data> input2{
        313709975877732223ULL,
        0ULL,
        0ULL
    };

     std::vector<Data> expected{
        431893979263004911ULL, 
        250850038660605336ULL, 
        117502165775093281ULL
    };



    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
    
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}