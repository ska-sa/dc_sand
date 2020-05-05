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
    m_ulSizeInputAntennaData(NR_STATIONS*NR_CHANNELS*NR_SAMPLES_PER_CHANNEL*2*sizeof(int8_t)),//Two is due to complex data
    m_ulSizeOutputBeamData(NR_BEAMS*NR_CHANNELS*NR_SAMPLES_PER_CHANNEL*2*sizeof(float)),
    m_eKernelOption(eKernelOption)
{   
    std::cout << m_ulSizeSteeringCoefficients/1000.0/1000.0 << " MB Allocated for steering coefficients" << std::endl;
    std::cout << m_ulSizeDelayValues/1000.0/1000.0 << " MB Allocated for delay values" << std::endl;
    std::cout << m_ulSizeInputAntennaData/1000.0/1000.0 << " MB Allocated for input antenna data" << std::endl;
    std::cout << m_ulSizeOutputBeamData/1000.0/1000.0 << " MB Allocated for output beam data" << std::endl;
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


    if(m_eKernelOption != COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL){
        GPU_ERRCHK(cudaMallocHost((void**)&m_pfHSteeringCoeffs,m_ulSizeSteeringCoefficients));
        GPU_ERRCHK(cudaMalloc((void**)&m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients));
    }else{
        GPU_ERRCHK(cudaMallocHost((void**)&m_pi8HInputAntennaData,m_ulSizeInputAntennaData));
        GPU_ERRCHK(cudaMalloc((void**)&m_pi8DInputAntennaData,m_ulSizeInputAntennaData));

        GPU_ERRCHK(cudaMallocHost((void**)&m_pfHOutputBeams,m_ulSizeOutputBeamData));
        GPU_ERRCHK(cudaMalloc((void**)&m_pfDOutputBeams,m_ulSizeOutputBeamData));
    }

    //Generating Grid and Block Sizes
    generate_GPU_kernel_dimensions();
}

BeamformerCoeffTest::~BeamformerCoeffTest()
{
    GPU_ERRCHK(cudaFree(m_pDDelayValues));
    GPU_ERRCHK(cudaFreeHost(m_pHDelayValues));
    if(m_eKernelOption != COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL){
        GPU_ERRCHK(cudaFree(m_pfDSteeringCoeffs));
        GPU_ERRCHK(cudaFreeHost(m_pfHSteeringCoeffs));
    }else{
        GPU_ERRCHK(cudaFree(m_pfDOutputBeams));
        GPU_ERRCHK(cudaFreeHost(m_pfHOutputBeams));
        GPU_ERRCHK(cudaFree(m_pi8DInputAntennaData));
        GPU_ERRCHK(cudaFreeHost(m_pi8HInputAntennaData));
    }
}

void BeamformerCoeffTest::generate_GPU_kernel_dimensions(){
    switch (m_eKernelOption)
    {
    //Refer to corresponding kernel functions for explanations as to how these blocks are generated
        case BeamformerCoeffTest::SteeringCoefficientKernel::NAIVE :
        {
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
        }
        break;

        case BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS :
        {
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
            int gridSizeChannels = NR_CHANNELS/NUM_CHANNELS_PER_KERNEL;
            if(NR_CHANNELS % NUM_CHANNELS_PER_KERNEL != 0){
                gridSizeChannels++;
            }
            m_cudaGridSize = dim3(numBlocksPerChannel,gridSizeChannels);
            m_cudaBlockSize = dim3(threadsPerBlock);
        }
        break;

        case BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS_AND_TIMESTAMPS :
        {
            size_t ulNumSamplesPerChannel = NR_STATIONS*NR_BEAMS;
            size_t ulNumBlocks = ulNumSamplesPerChannel/NUM_ANTBEAMS_PER_BLOCK;
            if(ulNumSamplesPerChannel%NUM_ANTBEAMS_PER_BLOCK != 0){
                ulNumBlocks++;
            }
            m_cudaGridSize = dim3(ulNumBlocks);//dim3(7,1);//
            m_cudaBlockSize = dim3(NUM_THREADS_PER_BLOCK_MAX);
            //std::cout << "Blocks: " << ulNumBlocks << std::endl;
            //std::cout << "Ants: " << NR_STATIONS << std::endl;
            //std::cout << "Beams: " << NR_BEAMS << std::endl;
            //std::cout << "Ants x Beams: " << NR_STATIONS*NR_BEAMS << std::endl;
        }

        case BeamformerCoeffTest::SteeringCoefficientKernel::COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL :
        {
            m_cudaGridSize = dim3(NR_CHANNELS);//dim3(7,1);//
            m_cudaBlockSize = dim3(NUM_THREADS_PER_BLOCK_MAX);
        }
        break;
    }    
}

void BeamformerCoeffTest::simulate_input()
{
    //Generates a delay value for every antenna-beam combination
    size_t ulNumDelayVelays = NR_STATIONS*NR_BEAMS;
    for (size_t i = 0; i < NR_STATIONS*NR_BEAMS; i++)
    {
        m_pHDelayValues[i].fDelay_s = 1;//((float)i/ulNumDelayVelays)*SAMPLING_PERIOD/3; //let's make them in a linear ramp
        m_pHDelayValues[i].fDelayRate_sps = 1;//2e-11;
        m_pHDelayValues[i].fPhase_rad = 1;//(1 -((float)i/ulNumDelayVelays))*SAMPLING_PERIOD/3;
        m_pHDelayValues[i].fPhaseRate_radps = 1;//3e-11;
    }

    if(m_eKernelOption == COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL){
        for (size_t i = 0; i < m_ulSizeInputAntennaData; i++)
        {
            m_pi8HInputAntennaData[i] = static_cast<int8_t>(i);
        }
    }
}

void BeamformerCoeffTest::transfer_HtoD()
{

    GPU_ERRCHK(cudaMemcpy(m_pDDelayValues,m_pHDelayValues,m_ulSizeDelayValues,cudaMemcpyHostToDevice));
    if(m_eKernelOption == COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL){
        GPU_ERRCHK(cudaMemcpy(m_pi8DInputAntennaData,m_pi8HInputAntennaData,m_ulSizeInputAntennaData,cudaMemcpyHostToDevice));
    }

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

        case BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS :
        for (size_t ulTimeIndex = 0; ulTimeIndex < NR_SAMPLES_PER_CHANNEL; ulTimeIndex++)
        {
            //Todo - subtract fix the tv_sec field when tv_ns goes above 999999999
            struct timespec sCurrentTime_ns;
            sCurrentTime_ns.tv_sec = m_sReferenceTime_ns.tv_sec;
            long lTimeStep = ulTimeIndex*SAMPLING_PERIOD*1e9*FFT_SIZE;
            sCurrentTime_ns.tv_nsec = m_sReferenceTime_ns.tv_nsec + lTimeStep;
            calculate_beamweights_grouped_channels<<<m_cudaGridSize,m_cudaBlockSize>>>(sCurrentTime_ns,m_sReferenceTime_ns,m_pHDelayValues,m_pfDSteeringCoeffs+NR_STATIONS*NR_CHANNELS*NR_BEAMS*2*ulTimeIndex);
        }
        break;

        case BeamformerCoeffTest::SteeringCoefficientKernel::MULTIPLE_CHANNELS_AND_TIMESTAMPS :
        {
            calculate_beamweights_grouped_channels_and_timestamps<<<m_cudaGridSize,m_cudaBlockSize>>>(m_sReferenceTime_ns,m_pHDelayValues,m_pfDSteeringCoeffs,false);
        }
        break;

        case BeamformerCoeffTest::SteeringCoefficientKernel::COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL :
        {
            calculate_beamweights_and_beamform_single_channel<<<m_cudaGridSize,m_cudaBlockSize>>>(m_sReferenceTime_ns,m_pHDelayValues,m_pfDOutputBeams,m_pi8HInputAntennaData);
        }
    }
}

void BeamformerCoeffTest::transfer_DtoH()
{
    if(m_eKernelOption != COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL)
    {
        GPU_ERRCHK(cudaMemcpy(m_pfHSteeringCoeffs,m_pfDSteeringCoeffs,m_ulSizeSteeringCoefficients,cudaMemcpyDeviceToHost));
    }
    else
    {
        GPU_ERRCHK(cudaMemcpy(m_pfHOutputBeams,m_pfDOutputBeams,m_ulSizeOutputBeamData,cudaMemcpyDeviceToHost));
    }
}

void BeamformerCoeffTest::verify_output()
{
    switch (m_eKernelOption)
    {
    case NAIVE:
    case MULTIPLE_CHANNELS:
    case MULTIPLE_CHANNELS_AND_TIMESTAMPS:
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
        break;

    case COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL:
        {
            int8_t * pi8InData = m_pi8HInputAntennaData;
            int8_t * pi8OutData = (int8_t*)m_pfHOutputBeams;
            for (size_t i = 0; i < m_ulSizeInputAntennaData; i++)
            {
                if (pi8InData[i] != pi8OutData[i])
                {
                    std::cout << i << " " << static_cast<int32_t>(pi8InData[i]) << " " << static_cast<int32_t>(pi8OutData[i]) <<  std::endl;
                    //m_iResult = -1;
                    return;
                }
            }
            m_iResult = 1;
            break;    
        }
    
    default:
        break;
    }
    
}

float BeamformerCoeffTest::get_time(){
    if(m_eKernelOption != COMBINED_COEFF_GEN_AND_BEAMFORMER_SINGLE_CHANNEL){
        float fRateOfFFTs_Hz = ((float)ADC_SAMPLE_RATE)/((float)FFT_SIZE);
        float fInputPacketSizePerFFT_Bytes = ((float)FFT_SIZE/2.0)/((float) NR_STATIONS) * NR_POLARIZATIONS * 2 * 8 * NR_STATIONS;
        float fTransferTimePerPacket_s = 1/fRateOfFFTs_Hz;
        m_fGpuUtilisation_SingleTimeUnit = (m_fKernelElapsedTime_ms/1000.0)/(NR_SAMPLES_PER_CHANNEL*fTransferTimePerPacket_s);
        m_fGpuUtilisation_MultipleTimeUnits = m_fGpuUtilisation_SingleTimeUnit/((float)ACCUMULATIONS_BEFORE_NEW_COEFFS);

        std::cout << "FFTs Per Second: " << fRateOfFFTs_Hz << " Hz" << std::endl;
        std::cout << "Size of X-Engine input per FFT: " << fInputPacketSizePerFFT_Bytes << " bytes" <<std::endl;
        std::cout << "Time to transfer a single packet: " << 1/fRateOfFFTs_Hz << " s" << std::endl;
        std::cout << "Time to transfer " << NR_SAMPLES_PER_CHANNEL << " packets: " << NR_SAMPLES_PER_CHANNEL/fRateOfFFTs_Hz << " s" << std::endl;
        std::cout << "Time to generate steering coefficients for " << NR_SAMPLES_PER_CHANNEL << " packets: " << m_fKernelElapsedTime_ms/1000.0 << " s" << std::endl;
        std::cout << std::endl;
        
        for (size_t i = 1; i < 4; i+=2)
        {
            std::cout << "With an X-Engine Ingest rate of " << fRateOfFFTs_Hz*fInputPacketSizePerFFT_Bytes/1e9f*(i+1) << " Gbps (" <<(i+1)<<" MeerKAT Polarisations)."<< std::endl;
            std::cout << "\tGPUs required with a 1:1 ratio of steering coefficients to input data: " << m_fGpuUtilisation_SingleTimeUnit*(i+1) << std::endl;
            std::cout << "\tGPUs required with a "<<ACCUMULATIONS_BEFORE_NEW_COEFFS<<":1 ratio of steering coefficients to input data: " << m_fGpuUtilisation_SingleTimeUnit/((float)ACCUMULATIONS_BEFORE_NEW_COEFFS)*(i+1) << std::endl;
            std::cout << std::endl;
        }

        m_fGpuUtilisation_SingleTimeUnit*=4; //Multiply by four, to equal two antennas worth of data
        m_fGpuUtilisation_MultipleTimeUnits*=4; //Multiply by four, to equal two antennas worth of data
    }else{
        m_fGpuUtilisation_SingleTimeUnit = m_fKernelElapsedTime_ms/m_fHtoDElapsedTime_ms;
        m_fGpuUtilisation_MultipleTimeUnits = m_fGpuUtilisation_SingleTimeUnit;
    }
    return UnitTest::get_time();
}

float BeamformerCoeffTest::get_gpu_utilisation_per_single_time_unit(){
    return m_fGpuUtilisation_SingleTimeUnit;
}

float BeamformerCoeffTest::get_gpu_utilisation_per_multiple_time_units(){
    return m_fGpuUtilisation_MultipleTimeUnits;
}