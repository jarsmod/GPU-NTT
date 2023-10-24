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

/**\
 * aS = np.array([3137099, 2439066, 5338084], dtype=np.uint64)
 * mdl = 576460752303415297
 * >>> (aS * 3137099) % mdl
 * array([ 9841390135801,  7651591509534, 16746097978316], dtype=uint64)
*/

TEST(SchoolBookPolyXply, smallervalues) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    //printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        3137099ULL,
        2439066ULL,
        5338084ULL
    };

    std::vector<Data> input2{
        3137099ULL,
        0ULL,
        0ULL
    };

     std::vector<Data> expected{
        9841390135801ULL, 
        7651591509534ULL, 
        16746097978316ULL
    };



    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
    
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}

/*
>>> aS = np.array([3137099, 2439066, 5338084], dtype=np.uint64)
>>> mdl = 576460752303415297
>>> px = np.polymul(aS, aS) % mdl
>>> if px.size % 2 == 1:
...     px = np.insert(px, px.size, 0)
>>> m = px.size // 2
>>> (px[:m] + px[m:]) % mdl # reduction by x-1
array([35881268514889, 43798323810124, 39441238908988], dtype=uint64)
*/

TEST(SchoolBookPolyXply, third) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    //printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        3137099ULL,
        2439066ULL,
        5338084ULL
    };


     std::vector<Data> expected{
        35881268514889ULL, 
        43798323810124ULL, 
        39441238908988ULL
    };



    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input1, parameters.modulus, ReductionPolynomial::X_N_minus);
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}

TEST(SchoolBookPolyXply, fourth) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    //printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        3137099ULL,
        2439066ULL,
        5338084ULL
    };

    std::vector<Data> input2{
        5338084ULL,
        2439066ULL,
        3137099ULL,
    };

     std::vector<Data> expected{
        37417628677394ULL, 
        37417628677394ULL, 
        44285573879213ULL,
    };



    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}

TEST(SchoolBookPolyXply, fifth) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    //printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        3137099, 2439066, 5338084, 2439066,
    };

    std::vector<Data> input2{
        5338084, 2439066, 3137099, 2439066,
    };

     std::vector<Data> expected{
        45390281861344, 41343061398156, 50234616831569, 41343061398156,
    };



    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}

/**
a = [31370993137099, 24390662439066, 53380845338084, 24390662439066]
b = [53380845338084, 24390662439066, 31370993137099, 24390662439066]

aS = np.array(a, dtype=np.uint64)
bS = np.array(b, dtype=np.uint64)

mdl = 576460752303415297

print(aS, bS, mdl)

px = np.polymul(aS, bS) % mdl

if px.size % 2 == 1:
    px = np.insert(px, px.size, 0)

m = px.size // 2
ans = (px[:m] + px[m:]) % mdl # reduction by x-1
print (ans)

*/
TEST(SchoolBookPolyXply, sixth) {
    
    
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    NTT_CPU generator(parameters);
        
    //printf("Parameters:\n\tmodulus:%llu, bit:%llu, mu:%llu\n", parameters.modulus.value, parameters.modulus.bit, parameters.modulus.mu);
    
    if(parameters.modulus.value != 576460752303415297){
        throw std::invalid_argument("modulus is not 576460752303415297");
    }

    std::vector<Data> input1{
        31370993137099, 24390662439066, 53380845338084, 24390662439066,
    };

    std::vector<Data> input2{
        53380845338084, 24390662439066, 31370993137099, 24390662439066,
    };

     std::vector<Data> expected{
        230352206369958105, 570020705986252373, 176229344153516850, 570020705985990261,
    };

    std::vector<Data> schoolbook_result 
        = schoolbook_poly_multiplication( input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);
     ASSERT_EQ( std::equal(expected.begin(), expected.end(), schoolbook_result.begin()), true );
    
}