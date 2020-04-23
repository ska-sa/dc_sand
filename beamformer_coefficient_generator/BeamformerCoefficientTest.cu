#include <cuComplex.h>
#include <cmath>
#include <iostream>
#include <chrono>

#include "BeamformerCoefficientTest.hpp"
#include "BeamformerKernels.cuh"
#include "Utils.hpp"
//#include "Kernels.cu"

float ts_diff(struct timespec first, struct timespec last)
// Give the difference between two timespecs, in floats. For opencl calculations.
{
    float time_difference = (float) last.tv_sec - (float) first.tv_sec;
    long nanosec_difference = last.tv_nsec - first.tv_nsec;
    time_difference += (float) nanosec_difference / 1e9f; //Should work if this is negative as well?
    return time_difference;
}

BeamformerCoeffTest::BeamformerCoeffTest(float fFloatingPointTolerance):
    m_fFloatingPointTolerance(fFloatingPointTolerance),
    m_ulSizeSteeringCoefficients(NR_SAMPLES_PER_CHANNEL*NR_CHANNELS * NR_STATIONS * NR_BEAMS * sizeof(cuFloatComplex)),
    m_ulSizeDelayValues(NR_STATIONS * NR_BEAMS * sizeof(struct delay_vals))
{   
    //Get timestamp of now
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

    //Generating Block Sizes
    int numSamplesPerChannel = NR_STATIONS*NR_BEAMS;
    int numBlocksPerChannel = numSamplesPerChannel/NUM_THREADS_PER_BLOCK;
    int threadsPerBlock = 0;
    if(numSamplesPerChannel%NUM_THREADS_PER_BLOCK != 0){
        numBlocksPerChannel++;
    }
    if(numBlocksPerChannel > 1){
        threadsPerBlock = NUM_THREADS_PER_BLOCK;
    }else{
        threadsPerBlock = numSamplesPerChannel;
    }
    m_cudaGridSize = dim3(numBlocksPerChannel,NR_STATIONS);//dim3(7,1);//
    m_cudaBlockSize = dim3(threadsPerBlock);
    //std::cout << "Block Size: " << threadsPerBlock << std::endl;
    //std::cout << "Grid Size: x: " << numBlocksPerChannel << " y: " << NR_STATIONS << std::endl; 

}

BeamformerCoeffTest::~BeamformerCoeffTest()
{
    GPU_ERRCHK(cudaFree(m_pDDelayValues));
    GPU_ERRCHK(cudaFreeHost(m_pHDelayValues));
    GPU_ERRCHK(cudaFree(m_pfDSteeringCoeffs));
    GPU_ERRCHK(cudaFreeHost(m_pfHSteeringCoeffs));
}

void BeamformerCoeffTest::simulate_input()
{
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
    GPU_ERRCHK(cudaMemcpy(m_pDDelayValues,m_pHDelayValues,m_ulSizeDelayValues,cudaMemcpyHostToDevice));
}

void BeamformerCoeffTest::run_kernel()
{
    for (size_t i = 0; i < NR_SAMPLES_PER_CHANNEL; i++)
    {
        //Todo - subtract fix the tv_sec field when tv_ns goes above 999999999
        struct timespec sCurrentTime_ns;
        sCurrentTime_ns.tv_sec = m_sReferenceTime_ns.tv_sec;
        long timeStep = i*SAMPLING_PERIOD*1e9*FFT_SIZE;
        sCurrentTime_ns.tv_nsec = m_sReferenceTime_ns.tv_nsec + timeStep;
        calculate_beamweights_naive<<<m_cudaGridSize,m_cudaBlockSize>>>(sCurrentTime_ns,m_sReferenceTime_ns,m_pHDelayValues,m_pfDSteeringCoeffs+NR_STATIONS*NR_CHANNELS*NR_BEAMS*2*i);   
    }
}

void BeamformerCoeffTest::transfer_DtoH()
{
    GPU_ERRCHK(cudaMemcpy(m_pfHSteeringCoeffs,m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients,cudaMemcpyDeviceToHost));
}

void BeamformerCoeffTest::verify_output()
{
    //int temp = 0;
    auto start = std::chrono::steady_clock::now();
    float * fCorrectDate = (float*)malloc(NR_SAMPLES_PER_CHANNEL*NR_BEAMS*NR_CHANNELS*NR_STATIONS*2*sizeof(float));
    for (size_t t = 0; t < NR_SAMPLES_PER_CHANNEL; t++)
    {
        //Todo - subtract fix the tv_sec field when tv_ns goes above 999999999
        struct timespec sCurrentTime_ns;
        sCurrentTime_ns.tv_sec = m_sReferenceTime_ns.tv_sec;
        long timeStep = t*SAMPLING_PERIOD*1e9*FFT_SIZE;
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
                    float fDelayN = (sDelayVal.fDelayRate_sps + fDeltaDelay)*c*M_PI/(SAMPLING_PERIOD*NR_CHANNELS);
                    float fDelayN2 = (sDelayVal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*M_PI/(SAMPLING_PERIOD*NR_CHANNELS);
                    float fDeltaPhase = sDelayVal.fPhaseRate_radps*fDeltaTime;
                    float fPhase0 = sDelayVal.fPhase_rad - fDelayN2 + fDeltaPhase;
                    float fRotation = fDelayN + fPhase0;
                    float fSteeringCoeffCorrectReal = cos(fRotation);//At least i think its the real one - may need to check this if its important
                    float fSteeringCoeffCorrectImag = sin(fRotation);

                    //Get data generated on GPU
                    size_t ulCoeffIndex =  2*(t*NR_CHANNELS*NR_STATIONS*NR_BEAMS+c*NR_STATIONS*NR_BEAMS + a*NR_BEAMS + b);
                    fCorrectDate[ulCoeffIndex] = fSteeringCoeffCorrectReal;
                    fCorrectDate[ulCoeffIndex+1] = fSteeringCoeffCorrectImag;

                    //if(ulCoeffIndex == 1680 /*|| (ulCoeffIndex > 1600 && ulCoeffIndex < 1700)*/ ){
                    //    std::cout << ulCoeffIndex << " C: " << c << ", A: " << a << ", B: " << b << " Correct data: " << fSteeringCoeffCorrectReal << " + " << fSteeringCoeffCorrectImag << "j" << std::endl;
                    //}
                }
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "CPU took " << (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << " ms to generate correct steering coefficients." << std::endl;

    //std::cout << NR_STATIONS*NR_CHANNELS*NR_BEAMS*2 << std::endl;
    for (size_t i = 0; i < NR_SAMPLES_PER_CHANNEL*NR_STATIONS*NR_CHANNELS*NR_BEAMS*2; i++)
    {
        //std::cout << "Index: " << i << ". Generated Value: " << m_pfHSteeringCoeffs[i] << ". Correct Value: " << fCorrectDate[i] << std::endl;
        if(std::abs(m_pfHSteeringCoeffs[i] - fCorrectDate[i]) > m_fFloatingPointTolerance){
            std::cout << "Index: " << i << ". Generated Value: " << m_pfHSteeringCoeffs[i] << ". Correct Value: " << fCorrectDate[i] << std::endl;
            // temp++;
            // if(temp == 1){
            m_iResult = -1;
            return;
            // }
        }
    }

    free(fCorrectDate);
    
    m_iResult = 1;
}