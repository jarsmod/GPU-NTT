
TEST(NTTParams, bitreverse) {
    //bitreverse(int index, int n_power)
    //basic
    ASSERT_EQ( bitreverse(0, 1) , 0);
    ASSERT_EQ( bitreverse(1, 1) , 1);
    
    // n is pow of 2
    ASSERT_EQ( bitreverse(0, 2) , 0);
    ASSERT_EQ( bitreverse(1, 2) , 2);
    ASSERT_EQ( bitreverse(2, 2) , 1);
    ASSERT_EQ( bitreverse(3, 2) , 3);

    //edge cases
    ASSERT_EQ(bitreverse(INT_MAX, INT_MAX), 0);
    ASSERT_EQ(bitreverse(INT_MAX, 1), 1);
    ASSERT_EQ(bitreverse(0, INT_MAX), 0);

    //random values for index and n_power
    ASSERT_EQ(bitreverse(7, 3), 7);
    ASSERT_EQ(bitreverse(10, 4), 5);
    ASSERT_EQ(bitreverse(22, 5), 13);

    //ASSERT_EQ(bitreverse(-22, 5), 13); // shouldn't happen

}