#include <iostream>     //For std::cout fucntionality
#include <iomanip>      //For setting widths and precision of data printed using std::cout
#include <tuple>        //For storing multiple types in a single array
#include <vector>       //Vector for storing tuples of results
#include <string> 

#include "BeamformerCoefficientTest.hpp"


int main()
{
    std::vector<std::tuple<std::string,float,float>> pResults(4);
    {
        std::cout << "Testing with a a kernel that generates steering coefficients and performs beamforming." << std::endl;
        BeamformerCoeffTest oBeamformerCoeffTest(1e-1f, BeamformerCoeffTest::SteeringCoefficientKernel::COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL);
        oBeamformerCoeffTest.run_test();
        oBeamformerCoeffTest.get_time();
        int iResult = oBeamformerCoeffTest.get_result();
        if(iResult!=1){
            std::cout << "Test failed, output data not generated correctly" << std::endl; 
            return 1;
        }
        pResults[3] = std::make_tuple("Combined Steering Coeffs+Beamforming",oBeamformerCoeffTest.get_gpu_utilisation_per_single_time_unit(), oBeamformerCoeffTest.get_gpu_utilisation_per_multiple_time_units());
    }

    {
        std::cout << "Testing with a single thread generating multiple steering coefficients(equal to the number of channels) per antenna-beam delay value." << std::endl;
        std::cout << "A single kernel generates multiple timestamps for a limited subset of delay values" << std::endl;
        BeamformerCoeffTest oBeamformerCoeffTest(1e-4f, BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS_AND_TIMESTAMPS);
        oBeamformerCoeffTest.run_test();
        oBeamformerCoeffTest.get_time();
        int iResult = oBeamformerCoeffTest.get_result();
        if(iResult!=1){
            std::cout << "Test failed, output data not generated correctly" << std::endl; 
            return 1;
        }
        pResults[2] = std::make_tuple("Multiple Chans+Timestamps",oBeamformerCoeffTest.get_gpu_utilisation_per_single_time_unit(), oBeamformerCoeffTest.get_gpu_utilisation_per_multiple_time_units());
    }


    {
        std::cout << "Testing with a single thread generating multiple steering coefficients(equal to the number of channels) per antenna-beam delay value for a single timestamp. A single kernel" << std::endl;
        std::cout << "A single kernel generates data for a single timestamp" << std::endl;
        BeamformerCoeffTest oBeamformerCoeffTest(1e-4f, BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS);
        oBeamformerCoeffTest.run_test();

        int iResult = oBeamformerCoeffTest.get_result();
        if(iResult!=1){
            std::cout << "Test failed, output data not generated correctly" << std::endl; 
            return 1;
        }
        oBeamformerCoeffTest.get_time();
        pResults[1] = std::make_tuple("Multiple Channels",oBeamformerCoeffTest.get_gpu_utilisation_per_single_time_unit(), oBeamformerCoeffTest.get_gpu_utilisation_per_multiple_time_units());
    }

    {
        std::cout << "Testing with a single thread generating a single steering coefficients per antenna-beam-channel delay value" << std::endl;
        BeamformerCoeffTest oBeamformerCoeffTest(1e-4f, BeamformerCoeffTest::SteeringCoefficientKernel::NAIVE);
        oBeamformerCoeffTest.run_test();
        
        int iResult = oBeamformerCoeffTest.get_result();
        if(iResult!=1){
            std::cout << "Test failed, output data not generated correctly" << std::endl; 
            return 1;
        }
        oBeamformerCoeffTest.get_time();
        pResults[0] = std::make_tuple("Naive Implementation",oBeamformerCoeffTest.get_gpu_utilisation_per_single_time_unit(), oBeamformerCoeffTest.get_gpu_utilisation_per_multiple_time_units());
    }

    std::cout << std::setw(50) << std::left << "Kernel Name" << std::setw(20)<< std::left << "GPU Utilisation" << std::setw(20) << std::left << "GPU Utilisation" << std::endl;
    std::cout << std::setw(50) << std::left << "" << std::setw(20)<< std::left << "(1 Time Unit)" << std::setw(20) << std::left << "(Many time Units)" << std::endl;
    for (size_t i = 0; i < 4; i++)
    {
       std::cout << std::setw(50) << std::left << std::get<0>(pResults[i]) << std::setw(20) << std::get<1>(pResults[i]) << std::setw(20) << std::get<2>(pResults[i]) << std::endl;
    }
    
    return 0;
}