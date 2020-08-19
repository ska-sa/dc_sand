#include <iostream>

#include "VectorAddTest.hpp"

int main()
{
    size_t uVectorLength = 833333333; // This will nearly fill a 2080 with ~11G of RAM. Don't attempt on a 2060.
    VectorAddTest myVectorAddTest(uVectorLength);
    myVectorAddTest.run_test();

    std::cout << myVectorAddTest.get_result() << "\n";
    myVectorAddTest.get_time();

    return 0;
}