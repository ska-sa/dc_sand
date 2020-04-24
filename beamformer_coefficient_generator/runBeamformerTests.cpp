#include <iostream>

#include "BeamformerCoefficientTest.hpp"

int main()
{
    
    BeamformerCoeffTest oBeamformerCoeffTest(1e-6f, BeamformerCoeffTest::SteeringCoefficientKernel::NAIVE);
    oBeamformerCoeffTest.run_test();
    
    std::cout << oBeamformerCoeffTest.get_result() << "\n";
    oBeamformerCoeffTest.get_time();

    return 0;
}