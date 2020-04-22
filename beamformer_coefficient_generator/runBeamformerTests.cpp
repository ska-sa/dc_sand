#include <iostream>

#include "BeamformerCoefficientTest.hpp"

int main()
{
    // This will nearly fill a 2080 with ~11G of RAM. Don't attempt on a 2060.
    BeamformerCoeffTest beamformerCoeffTest(1e-6);
    beamformerCoeffTest.run_test();
    
    std::cout << beamformerCoeffTest.get_result() << "\n";
    beamformerCoeffTest.get_time();

    return 0;
}