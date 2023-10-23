#include "helperutils.h"
#include <iostream>
#include <fstream>
#include <algorithm>

template <typename T>
void writeVectorToFile(const std::vector<T>& data, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    for (const T& value : data) {
        outputFile << value << "\n";
    }

    outputFile.close();
    std::cout << "Data has been written to " << filename << std::endl;
}

// Explicit instantiation for 'unsigned long long'
template void writeVectorToFile<unsigned long long>(const std::vector<unsigned long long>& data, const std::string& filename);
