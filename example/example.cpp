#include <iostream>

#include "VectorReduceComplexTest.hpp"

int main()
{
    // size_t uVectorLength = 833333333; // This will nearly fill a 2080 with ~11G of RAM. Don't attempt on a 2060.
    // Now working on dbelab07 with a 1080 GPU
    size_t uVectorLength = 1000;
    VectorReduceComplexTest myVectorReduceComplexTest(uVectorLength);
    myVectorReduceComplexTest.run_test();

    std::cout << myVectorReduceComplexTest.get_result() << "\n";
    myVectorReduceComplexTest.get_time();

    return 0;
}