#ifndef FILEWRITER_H
#define FILEWRITER_H

#include <vector>
#include <string>


template <typename T>
void writeVectorToFile(const std::vector<T>& data, const std::string& filename);

#endif
