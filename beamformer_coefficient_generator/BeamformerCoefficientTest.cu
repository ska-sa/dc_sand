#include <cuComplex.h>
#include <cmath>
#include <iostream>
#include <chrono>

#include "BeamformerCoefficientTest.hpp"
#include "BeamformerKernels.cuh"
#include "Utils.hpp"
//#include "Kernels.cu"

// Give the difference between two timespecs, in floats.
float ts_diff(struct timespec first, struct timespec last)
{
    float time_difference = (float) last.tv_sec - (float) first.tv_sec;
    long nanosec_difference = last.tv_nsec - first.tv_nsec;
    time_difference += (float) nanosec_difference / 1e9f; //Should work if this is negative as well?
    return time_difference;
}

BeamformerCoeffTest::BeamformerCoeffTest(float fFloatingPointTolerance, BeamformerCoeffTest::SteeringCoefficientKernel eKernelOption):
    m_fFloatingPointTolerance(fFloatingPointTolerance),
    m_ulSizeSteeringCoefficients(NR_SAMPLES_PER_CHANNEL*NR_CHANNELS * NR_STATIONS * NR_BEAMS * sizeof(cuFloatComplex)),
    m_ulSizeDelayValues(NR_STATIONS * NR_BEAMS * sizeof(struct delay_vals)),
    m_eKernelOption(eKernelOption)
{   
    std::cout << m_ulSizeSteeringCoefficients/1000.0/1000.0 << " MB Allocated" << std::endl;
    std::cout << m_ulSizeDelayValues/1000.0/1000.0 << " MB Allocated" << std::endl;
    //Generate a single reference time on initialisation
    struct timespec m_sReferenceTime_ns;
    clock_gettime(CLOCK_MONOTONIC, &m_sReferenceTime_ns);
     #define TIME_SHIFT  50000
     //Not quite sure what this is here for
    if (m_sReferenceTime_ns.tv_nsec >= TIME_SHIFT)
        m_sReferenceTime_ns.tv_nsec -= TIME_SHIFT;
    else
    {
        m_sReferenceTime_ns.tv_sec -= 1;
        m_sReferenceTime_ns.tv_nsec += (1000000000 - TIME_SHIFT);
    }
    
    //Initialising Memory
    GPU_ERRCHK(cudaMallocHost((void**)&m_pHDelayValues,m_ulSizeDelayValues));
    GPU_ERRCHK(cudaMalloc((void**)&m_pDDelayValues,m_ulSizeDelayValues));

    GPU_ERRCHK(cudaMallocHost((void**)&m_pfHSteeringCoeffs,m_ulSizeSteeringCoefficients));
    GPU_ERRCHK(cudaMalloc((void**)&m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients));

    //Generating Grid and Block Sizes
    generate_GPU_kernel_dimensions();
}

BeamformerCoeffTest::~BeamformerCoeffTest()
{
    GPU_ERRCHK(cudaFree(m_pDDelayValues));
    GPU_ERRCHK(cudaFreeHost(m_pHDelayValues));
    GPU_ERRCHK(cudaFree(m_pfDSteeringCoeffs));
    GPU_ERRCHK(cudaFreeHost(m_pfHSteeringCoeffs));
}

void BeamformerCoeffTest::generate_GPU_kernel_dimensions(){
    switch (m_eKernelOption)
    {
    //Refer to corresponding kernel functions for explanations as to how these blocks are generated
    case BeamformerCoeffTest::SteeringCoefficientKernel::NAIVE :
        size_t ulNumSamplesPerChannel = NR_STATIONS*NR_BEAMS;
        size_t ulNumBlocksPerChannel = ulNumSamplesPerChannel/NUM_THREADS_PER_BLOCK;
        size_t ulNumThreadsPerBlock = 0;
        if(ulNumSamplesPerChannel%NUM_THREADS_PER_BLOCK != 0){
            ulNumBlocksPerChannel++;
        }
        if(ulNumBlocksPerChannel > 1){
            ulNumThreadsPerBlock = NUM_THREADS_PER_BLOCK;
        }else{
            ulNumThreadsPerBlock = ulNumSamplesPerChannel;
        }
        m_cudaGridSize = dim3(ulNumBlocksPerChannel,NR_CHANNELS);//dim3(7,1);//
        m_cudaBlockSize = dim3(ulNumThreadsPerBlock);
        break;
    }
}

void BeamformerCoeffTest::simulate_input()
{
    //Generates a delay value for every antenna-beam combination
    size_t ulNumDelayVelays = NR_STATIONS*NR_BEAMS;
    for (size_t i = 0; i < NR_STATIONS*NR_BEAMS; i++)
    {
        m_pHDelayValues[i].fDelay_s = ((float)i/ulNumDelayVelays)*SAMPLING_PERIOD/3; //let's make them in a linear ramp
        m_pHDelayValues[i].fDelayRate_sps = 2e-11;
        m_pHDelayValues[i].fPhase_rad = (1 -((float)i/ulNumDelayVelays))*SAMPLING_PERIOD/3;
        m_pHDelayValues[i].fPhaseRate_radps = 3e-11;
    }
}

void BeamformerCoeffTest::transfer_HtoD()
{
    std::cout << "Transferring " << m_ulSizeDelayValues/1000.0/1000.0 << " MB to device" << std::endl;
    GPU_ERRCHK(cudaMemcpy(m_pDDelayValues,m_pHDelayValues,m_ulSizeDelayValues,cudaMemcpyHostToDevice));
}

void BeamformerCoeffTest::run_kernel()
{
    switch (m_eKernelOption)
    {
        case BeamformerCoeffTest::SteeringCoefficientKernel::NAIVE :
        for (size_t ulTimeIndex = 0; ulTimeIndex < NR_SAMPLES_PER_CHANNEL; ulTimeIndex++)
        {
            //Todo - subtract fix the tv_sec field when tv_ns goes above 999999999
            struct timespec sCurrentTime_ns;
            sCurrentTime_ns.tv_sec = m_sReferenceTime_ns.tv_sec;
            long lTimeStep = ulTimeIndex*SAMPLING_PERIOD*1e9*FFT_SIZE;
            sCurrentTime_ns.tv_nsec = m_sReferenceTime_ns.tv_nsec + lTimeStep;
            calculate_beamweights_naive<<<m_cudaGridSize,m_cudaBlockSize>>>(sCurrentTime_ns,m_sReferenceTime_ns,m_pHDelayValues,m_pfDSteeringCoeffs+NR_STATIONS*NR_CHANNELS*NR_BEAMS*2*ulTimeIndex);   
        }
        break;
    }
}

void BeamformerCoeffTest::transfer_DtoH()
{
    std::cout << "Transferring " << m_ulSizeSteeringCoefficients/1000.0/1000.0 << " MB to host" << std::endl;
    GPU_ERRCHK(cudaMemcpy(m_pfHSteeringCoeffs,m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients,cudaMemcpyDeviceToHost));
}

void BeamformerCoeffTest::verify_output()
{
    
    auto start = std::chrono::steady_clock::now();
    
    //Allocating matric to store correct data
    float * fCorrectDate = (float*)malloc(NR_SAMPLES_PER_CHANNEL*NR_BEAMS*NR_CHANNELS*NR_STATIONS*2*sizeof(float));
    
    //Generate correct data for all antennas, channels, timestamps and beam indices
    for (size_t t = 0; t < NR_SAMPLES_PER_CHANNEL; t++)
    {
        //Todo - subtract fix the tv_sec field when tv_ns goes above 999999999
        struct timespec sCurrentTime_ns;
        sCurrentTime_ns.tv_sec = m_sReferenceTime_ns.tv_sec;
        long timeStep = t*SAMPLING_PERIOD*1e9f*FFT_SIZE;
        sCurrentTime_ns.tv_nsec = m_sReferenceTime_ns.tv_nsec + timeStep;
        for (size_t c = 0; c < NR_CHANNELS; c++)
        {
            for (size_t a = 0; a < NR_STATIONS; a++)
            {
                for (size_t b = 0; b < NR_BEAMS; b++)
                {   
                    //Generate simulated data
                    struct delay_vals sDelayVal = m_pHDelayValues[a*NR_BEAMS + b];
                    float fDeltaTime = ts_diff(m_sReferenceTime_ns, sCurrentTime_ns);
                    float fDeltaDelay = sDelayVal.fDelayRate_sps*fDeltaTime;
                    float fDelayN = (sDelayVal.fDelayRate_sps + fDeltaDelay)*c*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
                    float fDelayN2 = (sDelayVal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
                    float fDeltaPhase = sDelayVal.fPhaseRate_radps*fDeltaTime;
                    float fPhase0 = sDelayVal.fPhase_rad - fDelayN2 + fDeltaPhase;
                    float fRotation = fDelayN + fPhase0;
                    float fSteeringCoeffCorrectReal = cos(fRotation);//At least i think its the real one - may need to check this if its important
                    float fSteeringCoeffCorrectImag = sin(fRotation);

                    //Write steering coefficient to the array
                    size_t ulCoeffIndex =  2*(t*NR_CHANNELS*NR_STATIONS*NR_BEAMS+c*NR_STATIONS*NR_BEAMS + a*NR_BEAMS + b);
                    fCorrectDate[ulCoeffIndex] = fSteeringCoeffCorrectReal;
                    fCorrectDate[ulCoeffIndex+1] = fSteeringCoeffCorrectImag;

                }
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "CPU took " << (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << " ms to generate correct steering coefficients." << std::endl;

    //int temp = 0;
    //Compare correct data to GPU generated data - this loop could be combined with the above loop, however this way makes it easier to time the CPU execution and add extra debug data if necessary
    for (size_t i = 0; i < NR_SAMPLES_PER_CHANNEL*NR_STATIONS*NR_CHANNELS*NR_BEAMS*2; i++)
    {
        if(std::abs(m_pfHSteeringCoeffs[i] - fCorrectDate[i]) > m_fFloatingPointTolerance /*|| (i > 49950)*/){
            std::cout << "Index: " << i << ". Generated Value: " << m_pfHSteeringCoeffs[i] << ". Correct Value: " << fCorrectDate[i] << std::endl;
            //if(temp++ == 100){
                m_iResult = -1;
                return;
            //}
        }
    }

    free(fCorrectDate);
    
    m_iResult = 1;
}