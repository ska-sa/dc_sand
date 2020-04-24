#ifndef __BEAMFORMERCOEFFICIENT_TEST_HPP__
#define __BEAMFORMERCOEFFICIENT_TEST_HPP__

#include "BeamformerParameters.h"

#include "UnitTest.hpp"
#include <cstdint>
#include <ctime>

class BeamformerCoeffTest : public UnitTest
{
    public:
        /** This enum is used on initialisation of the BeamformerCoeffTest object. It allows the 
         *  user to determine what kernel implementation of the beamformer to use. This is ueful
         *  for testing purposes.
         */
        enum SteeringCoefficientKernel
        {
            NAIVE
        };

        BeamformerCoeffTest(float fFloatingPointTolerance, SteeringCoefficientKernel eKernelOption);
        ~BeamformerCoeffTest();

    protected:
        void simulate_input() override;

        void transfer_HtoD() override;

        void run_kernel() override;

        void transfer_DtoH() override;

        void verify_output() override;

    private:
        //Level of tolerance to use when checking beamformer values equal correct values
        float m_fFloatingPointTolerance;

        //Host pointers
        delay_vals *m_pHDelayValues;
        float *m_pfHSteeringCoeffs;

        //device pointers
        delay_vals *m_pDDelayValues;
        float *m_pfDSteeringCoeffs;

        //Kernel Dimenstions
        dim3 m_cudaGridSize;
        dim3 m_cudaBlockSize;

        //Sizes of data to transfers
        size_t m_ulSizeSteeringCoefficients;
        size_t m_ulSizeDelayValues;

        //Delay rate specific values
        struct timespec m_sReferenceTime_ns;

        //This stores the kernel that will be executed
        SteeringCoefficientKernel  m_eKernelOption;

        ///Generates the kernel dimensions. This is called in the constructor and the only reason that it is a seperate function is to keep the constructor clean. 
        void generate_GPU_kernel_dimensions();
};

#endif