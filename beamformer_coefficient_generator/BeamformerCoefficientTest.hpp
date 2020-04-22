#ifndef __BEAMFORMERCOEFFICIENT_TEST_HPP__
#define __BEAMFORMERCOEFFICIENT_TEST_HPP__

#include "BeamformerParameters.h"

#include "UnitTest.hpp"
#include <cstdint>
#include <ctime>

class BeamformerCoeffTest : public UnitTest
{
    public:
        BeamformerCoeffTest(float fFloatingPointTolerance);
        ~BeamformerCoeffTest();

    protected:
        void simulate_input() override;

        void transfer_HtoD() override;

        void run_kernel() override;

        void transfer_DtoH() override;

        void verify_output() override;

    private:
        struct delay_vals_extended {
            float fSamplingPeriod_s; // seconds. Inverse of sampling frequency.
            struct timespec sRefTime_ns; //epoch time, nanosecond precision
            float fDelay_s; // seconds
            float fDelayRate_sps; //seconds per second
            float fPhase_rad; //radians
            float fPhaseRate_radps; //radians per second
        };

        //Level of tolerance to use when checking beamformer values to correct values
        float m_fFloatingPointTolerance;

        //Host pointers
        delay_vals_extended *m_pHDelayValues;
        float *m_pfHSteeringCoeffs;

        //device pointers
        delay_vals_extended *m_pDDelayValues;
        float *m_pfDSteeringCoeffs;

        //Kernel Dimenstions
        dim3 m_cudaGridSize;
        dim3 m_cudaBlockSize;

        //Sizes of data to transfers
        size_t m_ulSizeSteeringCoefficients;
        size_t m_ulSizeDelayValues;

        //Delay rate specific values
        struct timespec m_sCurrentTime_ns;
};

#endif