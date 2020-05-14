#ifndef __BEAMFORMERCOEFFICIENT_TEST_HPP__
#define __BEAMFORMERCOEFFICIENT_TEST_HPP__

#include "BeamformerParameters.h"

#include "UnitTest.hpp"
#include <cstdint>
#include <ctime>

class BeamformerCoeffTest : public UnitTest
{
    public:
        /** \brief      Used to specify the GPU kernel that will generate the steering coefficients 
         * 
         *  \details    This enum is used on initialisation of the BeamformerCoeffTest object. It allows the 
         *              user to determine what kernel implementation of the beamformer to use. This is ueful
         *              for testing purposes.
         */
        enum SteeringCoefficientKernel
        {
            NAIVE,
            MULTIPLE_CHANNELS,
            MULTIPLE_CHANNELS_AND_TIMESTAMPS,
            COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL
        };

        enum SteeringCoefficientBitWidth
        {
            b16,
            b32
        };

        BeamformerCoeffTest(float fFloatingPointTolerance, SteeringCoefficientKernel eKernelOption, SteeringCoefficientBitWidth eBitWidth);
        ~BeamformerCoeffTest();

        /** \brief Overriden function to calculate GPU utilisation of steering coefficient generation
         * 
         *  \details    This function is overriden  as the steering coefficients are transferred to the GPU at a very slow rate.
         *              This means that the simple compute_time/pcie_transfer time calculation wont produce anything useful.
         *              This functiona also reports the GPU utilisation for an increasing number of MeerKAT pols. This is useful 
         *              for mapping number of antennas to the number of GPUs. 
         */ 
        float get_time() override;

        float get_gpu_utilisation_per_single_time_unit();

        float get_gpu_utilisation_per_multiple_time_units();

    protected:
        void simulate_input() override;

        void transfer_HtoD() override;

        void run_kernel() override;

        void transfer_DtoH() override;

        void verify_output() override;

    private:
        //Level of tolerance to use when checking beamformer values equal correct values
        float m_fFloatingPointTolerance;

        //Host pointers Delay Values Generation
        delay_vals *m_pHDelayValues;
        float *m_pfHSteeringCoeffs;

        //Device pointers Delay Value Generation
        delay_vals *m_pDDelayValues;
        float *m_pfDSteeringCoeffs;

        //Host pointers beamformer
        int8_t *m_pi8HInputAntennaData;
        float *m_pfHOutputBeams;

        //Device pointers delay value generation
        int8_t *m_pi8DInputAntennaData;
        float *m_pfDOutputBeams;

        //Kernel Dimenstions
        dim3 m_cudaGridSize;
        dim3 m_cudaBlockSize;

        //Sizes of data to transfer
        size_t m_ulSizeSteeringCoefficients;
        size_t m_ulSizeDelayValues;
        size_t m_ulSizeInputAntennaData;
        size_t m_ulSizeOutputBeamData;


        //Delay rate specific values
        struct timespec m_sReferenceTime_ns;

        //GPU Utilisation for various lengths of time. Calculated in get_time() method
        float m_fGpuUtilisation_SingleTimeUnit;
        float m_fGpuUtilisation_MultipleTimeUnits;

        //This stores the kernel that will be executed
        SteeringCoefficientKernel  m_eKernelOption;

        //This stores the number of bits that will be generated
        SteeringCoefficientBitWidth m_eBitWidth;

        ///Generates the kernel dimensions. This is called in the constructor and the only reason that it is a seperate function is to keep the constructor clean. 
        void generate_GPU_kernel_dimensions();

};

#endif