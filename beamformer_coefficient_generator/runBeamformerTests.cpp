#include <iostream>

#include "BeamformerCoefficientTest.hpp"

int main()
{
    
    BeamformerCoeffTest beamformerCoeffTest(1e-6f);
    beamformerCoeffTest.run_test();
    
    std::cout << beamformerCoeffTest.get_result() << "\n";
    beamformerCoeffTest.get_time();

    return 0;
}