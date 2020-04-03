#include <iostream>

#include "VectorAddTest.hpp"

int main()
{
    size_t uVectorLength = 1000000000; // 1G.
    VectorAddTest myVectorAddTest(uVectorLength);
    myVectorAddTest.run_test();

    return 0;
}