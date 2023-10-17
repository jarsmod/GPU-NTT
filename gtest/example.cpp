#include <vector>
#include <gtest/gtest.h>

/**
$ sudo apt-get install libgtest-dev
$ g++ test.cpp -lgtest
*/

// Function to multiply a vector of integers by a constant
std::vector<int> multiplyVectorByConstant(const std::vector<int>& vector, int constant) {
    std::vector<int> result;
    for (const int& element : vector) {
        result.push_back(element * constant);
    }
    return result;
}

TEST(VectorMultiplicationTest, MultiplyByZero) {
    std::vector<int> vector = {1, 2, 3, 4};
    int constant = 0;
    std::vector<int> result = multiplyVectorByConstant(vector, constant);
    std::vector<int> expectedResult = {0, 0, 0, 0};
    ASSERT_EQ(result, expectedResult);
}

TEST(VectorMultiplicationTest, MultiplyByPositiveConstant) {
    std::vector<int> vector = {1, 2, 3, 4};
    int constant = 2;
    std::vector<int> result = multiplyVectorByConstant(vector, constant);
    std::vector<int> expectedResult = {2, 4, 6, 8};
    ASSERT_EQ(result, expectedResult);
}

TEST(VectorMultiplicationTest, MultiplyByNegativeConstant) {
    std::vector<int> vector = {1, 2, 3, 4};
    int constant = -2;
    std::vector<int> result = multiplyVectorByConstant(vector, constant);
    std::vector<int> expectedResult = {-2, -4, -6, -8};
    ASSERT_EQ(result, expectedResult);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



