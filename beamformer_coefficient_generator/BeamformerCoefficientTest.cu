#include <cuComplex.h>
#include <cmath>
#include <iostream>

#include "BeamformerCoefficientTest.hpp"
#include "BeamformerKernels.cuh"
#include "Utils.hpp"
//#include "Kernels.cu"

float ts_diff(struct timespec first, struct timespec last)
// Give the difference between two timespecs, in floats. For opencl calculations.
{
    float time_difference = (float) last.tv_sec - (float) first.tv_sec;
    long nanosec_difference = last.tv_nsec - first.tv_nsec;
    time_difference += (float) nanosec_difference / 1e9; //Should work if this is negative as well?
    return time_difference;
}

BeamformerCoeffTest::BeamformerCoeffTest(float fFloatingPointTolerance):
    m_fFloatingPointTolerance(fFloatingPointTolerance),
    m_ulSizeSteeringCoefficients(NR_CHANNELS * NR_STATIONS * NR_BEAMS * sizeof(cuFloatComplex)),
    m_ulSizeDelayValues(NR_STATIONS * NR_BEAMS * sizeof(struct delay_vals_extended))
{   
    //Get timestamp of now
    struct timespec m_sCurrentTime_ns;
    clock_gettime(CLOCK_MONOTONIC, &m_sCurrentTime_ns);
    
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
    m_cudaGridSize = dim3(numBlocksPerChannel,NR_STATIONS);
    m_cudaBlockSize = dim3(threadsPerBlock);
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
    float fSamplingPeriod = 1e-9;
    struct timespec sRefTime;
    sRefTime.tv_sec = m_sCurrentTime_ns.tv_sec;
    sRefTime.tv_nsec = m_sCurrentTime_ns.tv_nsec;

    #define TIME_SHIFT  50000
    if (sRefTime.tv_nsec >= TIME_SHIFT)
        sRefTime.tv_nsec -= TIME_SHIFT;
    else
    {
        sRefTime.tv_sec -= 1;
        sRefTime.tv_nsec += (1000000000 - TIME_SHIFT);
    }

    size_t ulNumDelayVelays = NR_STATIONS*NR_BEAMS;
    for (size_t i = 0; i < NR_STATIONS*NR_BEAMS; i++)
    {
        m_pHDelayValues[i].fSamplingPeriod_s = fSamplingPeriod;
        m_pHDelayValues[i].sRefTime_ns = sRefTime;
        m_pHDelayValues[i].fDelay_s = ((float)i/ulNumDelayVelays)*fSamplingPeriod/3; //let's make them in a linear ramp
        m_pHDelayValues[i].fDelayRate_sps = 2e-11;
        m_pHDelayValues[i].fPhase_rad = (1 -((float)i/ulNumDelayVelays))*fSamplingPeriod/3;
        m_pHDelayValues[i].fPhaseRate_radps = 3e-11;
    }
}

void BeamformerCoeffTest::transfer_HtoD()
{
    GPU_ERRCHK(cudaMemcpy(m_pDDelayValues,m_pHDelayValues,m_ulSizeDelayValues,cudaMemcpyHostToDevice));
}

void BeamformerCoeffTest::run_kernel()
{
    calculate_beamweights_naive<<<m_cudaGridSize,m_cudaBlockSize>>>(m_sCurrentTime_ns,m_pHDelayValues,m_pfDSteeringCoeffs);
}

void BeamformerCoeffTest::transfer_DtoH()
{
    GPU_ERRCHK(cudaMemcpy(m_pfHSteeringCoeffs,m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients,cudaMemcpyDeviceToHost));
}

void BeamformerCoeffTest::verify_output()
{
    float * fCorrectDate = (float*)malloc(NR_BEAMS*NR_CHANNELS*NR_STATIONS*2*sizeof(float));
    for (size_t c = 0; c < NR_CHANNELS; c++)
    {
        for (size_t a = 0; a < NR_STATIONS; a++)
        {
            for (size_t b = 0; b < NR_BEAMS; b++)
            {   
                //Generate simulated data
                struct delay_vals_extended sDelayVal = m_pHDelayValues[a*NR_BEAMS + b];
                float fDeltaTime = ts_diff(sDelayVal.sRefTime_ns, m_sCurrentTime_ns);
                float fDeltaDelay = sDelayVal.fDelayRate_sps*fDeltaTime;
                float fDelayN = (sDelayVal.fDelayRate_sps + fDeltaDelay)*c*M_PI/(sDelayVal.fSamplingPeriod_s*NR_CHANNELS);
                float fDelayN2 = (sDelayVal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*M_PI/(sDelayVal.fSamplingPeriod_s*NR_CHANNELS);
                float fDeltaPhase = sDelayVal.fPhaseRate_radps*fDeltaTime;
                float fPhase0 = sDelayVal.fPhase_rad - fDelayN2 + fDeltaPhase;
                float fRotation = fDelayN + fPhase0;
                float fSteeringCoeffCorrectReal = cos(fRotation);//At least i think its the real one - may need to check this if its important
                float fSteeringCoeffCorrectImag = sin(fRotation);

                //Get data generated on GPU
                size_t ulCoeffIndex =  2*(c*NR_STATIONS*NR_BEAMS + a*NR_BEAMS + b);
                fCorrectDate[ulCoeffIndex] = fSteeringCoeffCorrectReal;
                fCorrectDate[ulCoeffIndex+1] = fSteeringCoeffCorrectImag;
                //float fSteeringCoeffGeneratedReal = m_pfHSteeringCoeffs[ulCoeffIndex];
                //float fSteeringCoeffGeneratedImag = m_pfHSteeringCoeffs[ulCoeffIndex+1];

                //std::cout << ulCoeffIndex << std::endl;
                //if(std::abs(fSteeringCoeffGeneratedReal - fSteeringCoeffCorrectReal) > m_fFloatingPointTolerance
                //    || std::abs(fSteeringCoeffGeneratedImag - fSteeringCoeffCorrectImag) > m_fFloatingPointTolerance)
                //{
                //    std::cout << fSteeringCoeffGeneratedReal << " " <<  fSteeringCoeffCorrectReal << std::endl;
                //    std::cout << fSteeringCoeffGeneratedImag << " " <<  fSteeringCoeffCorrectImag << std::endl;
                //    m_iResult = -1;
                //    //return;
                //}

                //cplx_beamweights[2*(c*n_antennas*n_beams + a*n_beams + b)] = cos(rotation);
                //cplx_beamweights[2*(c*n_antennas*n_beams + a*n_beams + b)+1] = sin(rotation);
                //std::cout << " c: " << c << " b: " << b << " a: " << a << " r: " << cplx_beamweights[2*(c*n_antennas*n_beams + a*n_beams + b)] << " i: " << cplx_beamweights[2*(c*n_antennas*n_beams + a*n_beams + b)+1] <<std::endl; 
            }
        }
    }

    std::cout << NR_STATIONS*NR_CHANNELS*NR_BEAMS*2 << std::endl;
    for (size_t i = 0; i < NR_STATIONS*NR_CHANNELS*NR_BEAMS*2; i++)
    {
        if(std::abs(m_pfHSteeringCoeffs[i] - fCorrectDate[i]) > m_fFloatingPointTolerance){
            std::cout << i << " " << m_pfHSteeringCoeffs[i] << " " << fCorrectDate[i] << std::endl;
            //m_iResult = -1;
            //return;
        }
    }

    free(fCorrectDate);
    
    m_iResult = 1;
}