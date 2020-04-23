#ifndef __BEAMFORMERCOEFFICIENT_TEST_HPP__
#define __BEAMFORMERCOEFFICIENT_TEST_HPP__

#include "BeamformerParameters.h"

#include "UnitTest.hpp"
#include <cstdint>
#include <ctime>

class BeamformerCoeffTest : public UnitTest
{
    public:
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
        //Level of tolerance to use when checking beamformer values to correct values
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

        //Kernel to use
        SteeringCoefficientKernel  m_eKernelOption;

        void generate_GPU_kernel_dimensions();
};

#endif