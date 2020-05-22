#include <stdio.h> 
#include "BeamformerParameters.h"
#include "cuComplex.h"
#include <cstdint>
#include "cuda_fp16.h"

__global__ void calculate_beamweights_naive(
                                struct timespec sCurrentTime, 
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs)
{
    int iInterChannelIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(iInterChannelIndex < NR_BEAMS*NR_STATIONS)
    {
        //***** Determine Correct Indices *****
        int iChannelIndex = blockIdx.y; 
        int iAntIndex = iInterChannelIndex/NR_BEAMS;
        int iBeamIndex = iInterChannelIndex - iAntIndex * NR_BEAMS;

        //***** Calculate steering coefficient values ****
        struct delay_vals sDelayValuesLocal = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];

        //TODO: Add logic to detect if nanoseconds go above 999999999 and increments seconds by 1
        float fTimeDifference = sCurrentTime.tv_sec - sRefTime.tv_sec;
        long fNanosecondsTimeDifference = sCurrentTime.tv_nsec - sRefTime.tv_nsec;
        fTimeDifference += (float) fNanosecondsTimeDifference / 1e9f;

        float fDeltaTime = fTimeDifference;
        float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
        float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
        float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
        float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
        float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
        float fRotation = fDelayN + fPhase0;
        float fSteeringCoeffCorrectReal;
        float fSteeringCoeffCorrectImag;

        /** cuda function to combine cosf and sinf into a single operation
         *  The __ means that a faster less precise lookup is being performed, 
         *  we will eventually need to determin if this is accurate enough for 
         *  our purposes
         */
        __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
        
        //***** Write generated steering coefficients to global memory *****
        size_t ulOutputIndex = (iChannelIndex*NR_STATIONS*NR_BEAMS + iInterChannelIndex)*2;
        pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;
        pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
    }

}



__global__ void calculate_beamweights_grouped_channels(
                                struct timespec sCurrentTime, 
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs,
                                bool b16BitOutput)
{
    int iInterChannelIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(iInterChannelIndex < NR_BEAMS*NR_STATIONS)
    {
        //***** Calculate all relevant indexes *****
        int iAntIndex = iInterChannelIndex/NR_BEAMS;
        int iBeamIndex = iInterChannelIndex - iAntIndex * NR_BEAMS;
        //Calculate which channels to iterate through
        int iChannelBlockIndex = blockIdx.y;
        int iChannelStartIndex = iChannelBlockIndex*NUM_CHANNELS_PER_KERNEL;
        int iChannelStopIndex = (iChannelBlockIndex+1)*(NUM_CHANNELS_PER_KERNEL);
        if(iChannelStartIndex > NR_CHANNELS)
        {
            iChannelStopIndex = NR_CHANNELS;
        }

        //***** Load delay values from global memory
        struct delay_vals sDelayValuesLocal = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];

        //***** Calculate all steering coefficients for each channel of a single antenna pair.
        //TODO: Add logic to detect if nanoseconds go above 999999999 and increments seconds by 1
        float fDeltaTime = sCurrentTime.tv_sec - sRefTime.tv_sec;
        long fNanosecondsTimeDifference = sCurrentTime.tv_nsec - sRefTime.tv_nsec;
        fDeltaTime += (float) fNanosecondsTimeDifference / 1e9f;

        float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
        float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
        float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);

        size_t ulOutputIndex;
        //Iterate through all required channels
        for(int iChannelIndex = iChannelStartIndex; iChannelIndex < iChannelStopIndex; iChannelIndex++)
        {
            float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
            float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
            float fRotation = fDelayN + fPhase0;
            float fSteeringCoeffCorrectReal;
            float fSteeringCoeffCorrectImag;
            sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
            ulOutputIndex = (iChannelIndex*NR_STATIONS*NR_BEAMS + iInterChannelIndex);
            
            //***** Store calculated values back into global memory *****
            //Casts to 16 bit if this is necessary.
            if(!b16BitOutput)
            {
                ulOutputIndex*=2;
                pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;
                pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
            }
            else
            {
                __half2 h2PackedOutput = __floats2half2_rn(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                //printf("Orig %f Converted %f %f\n",fSteeringCoeffCorrectReal,__high2float(h2PackedOutput), __low2float(h2PackedOutput)); \\Leaving here uncommented, useful for debugging
                ((__half2*)pfCplxSteeringCoeffs)[ulOutputIndex] = h2PackedOutput;
            }
        }
    }
}

__global__ void calculate_beamweights_grouped_channels_and_timestamps(
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs,
                                bool b16BitOutput)
{
    //Calculate Relevant Indices
    int iBeamAntIndex = blockIdx.x*NUM_ANTBEAMS_PER_BLOCK + threadIdx.x % NUM_ANTBEAMS_PER_BLOCK;
    
    //Create shared memory array to store a subset of the delay values
    __shared__ struct delay_vals psSDelayVals[NUM_ANTBEAMS_PER_BLOCK];
    
    //***** Read delay values from global memory into shared memory *****
    //More efficient to load psDelayVals as four bytes per thread instead of 16 
    //bytes per thread, this is why the array is cast to int32_t.
    if(threadIdx.x<NUM_ANTBEAMS_PER_BLOCK*4)
    {
        int iStartOffset =  blockIdx.x*NUM_ANTBEAMS_PER_BLOCK*4;
        ((int32_t *) psSDelayVals)[threadIdx.x] = ((int32_t *)psDelayVals)[(iStartOffset + threadIdx.x)];
    }

    __syncthreads();
    //***** Calculate Steering Coefficients *****
    if(iBeamAntIndex < NR_BEAMS*NR_STATIONS)
    {
        struct delay_vals sDelayValuesLocal = psSDelayVals[threadIdx.x % NUM_ANTBEAMS_PER_BLOCK];
        // Each thread geenerates a subset of steering coefficients time samples for a single delay value
        // Multiple threads together generate all the steering coefficients per time sample 
        const int iTotalTimeChunksPerKernel = NUM_THREADS_PER_BLOCK_MAX/(NUM_ANTBEAMS_PER_BLOCK*2);//This multiply by two is here to increase the width of a single time chunk. This will better coalesce the writing of data to the pfCplxSteeringCoeffs. After some tweaking, 2 works best for 32-bit floating point outputs and 1 for 16-bit floating point outputs. When this is changed, the multiply by two in iTimeChunk two lines down must also be changed. There is probably a better way to do this but I have not put much thought into it.
        const int iTimeIterations = NR_SAMPLES_PER_CHANNEL/iTotalTimeChunksPerKernel;
        int iTimeChunk = threadIdx.x / (NUM_ANTBEAMS_PER_BLOCK*2);
        #pragma unroll
        for (int iTimeStep = 0; iTimeStep < iTimeIterations; iTimeStep++)
        {
            int iTimeIndex = iTimeChunk + iTimeStep*iTotalTimeChunksPerKernel;
            float fDeltaTime = iTimeIndex*SAMPLING_PERIOD*FFT_SIZE;
            float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
            float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
            float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);

            size_t ulOutputIndex;
            //Iterate through all required channels
            for(int iChannelIndex = 0; iChannelIndex < NR_CHANNELS; iChannelIndex++)
            {
                float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
                float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
                float fRotation = fDelayN + fPhase0;
                float fSteeringCoeffCorrectReal;
                float fSteeringCoeffCorrectImag;
                __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);

                if(!b16BitOutput)
                {
                    //32 bit output
                    ulOutputIndex = (NR_STATIONS*NR_CHANNELS*NR_BEAMS*iTimeIndex + iChannelIndex*NR_STATIONS*NR_BEAMS + iBeamAntIndex)*2;
                    pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;
                    pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
                }
                else
                {
                    ulOutputIndex = (NR_STATIONS*NR_CHANNELS*NR_BEAMS*iTimeIndex + iChannelIndex*NR_STATIONS*NR_BEAMS + iBeamAntIndex);
                    __half2 h2PackedOutput = __floats2half2_rn(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                    //printf("Orig %f Converted %f %f\n",fSteeringCoeffCorrectReal,__high2float(h2PackedOutput), __low2float(h2PackedOutput)); //Going to leave this statement here for debugging - it is useful
                    ((__half2*)pfCplxSteeringCoeffs)[ulOutputIndex] = h2PackedOutput;
                }
            }
        }
    }
}

//TODO Iterate over a number of channels to reduce the shared memory usage
__global__ void calculate_beamweights_and_beamform_single_channel(
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfBeams,
                                int8_t * pi8AntennaData)
{   
    /** This array stores the all the delay values. I do not think that it is 
     *  strictly necessary as each thread only accesses a single delay value. I 
     *  have done this is a find loading 4 byte chunk per thread is faster than
     *  loading 16 bytes per thread - the shared memory array allows the loading
     *  of data by multiple threads and then accessing by 1 thread efficiently. 
     *  Figuring out a better method is probably better
     */
    __shared__ delay_vals psDelayValsShared[NR_STATIONS*NR_BEAMS];

    //Stores a single channels worth of F-Engine output data
    __shared__ int8_t pi8AntennaDataInShared[NR_STATIONS][INTERNAL_TIME_SAMPLES][COMPLEXITY];
    /** This struct will hold the output value of the warp level reduction 
     *  operation that will occur later in the kernel. This is needed as the 
     *  sums of the warps need to to be added together due to the number of 
     *  antennas being greater than the number of warps.
     *  
     *  The divide by 32 is beacause each set of 32 station is processed as a 
     *  single warp and added together
     */
    __shared__ float warpSums[NR_BEAMS][INTERNAL_TIME_SAMPLES][NR_STATIONS/32][COMPLEXITY];
    
    int iThreadIndex = threadIdx.x;
    int iChannelIndex = blockIdx.x;

    //***** Load delay values into memory *****
    const int iNumDelayValueTransfersTotal_32bitWords = NR_BEAMS*NR_STATIONS*sizeof(struct delay_vals)/sizeof(int32_t);
    const int iNumDelayValueTransfersPerThread = iNumDelayValueTransfersTotal_32bitWords/NUM_THREADS_PER_BLOCK_MAX;
    const int iOffsetBetweenDelayValueLoopIterations_32bitWords = NUM_THREADS_PER_BLOCK_MAX;
    #pragma unroll
    for (size_t i = 0; i < iNumDelayValueTransfersPerThread; i++)
    {
        int iGlobalMemoryIndex = threadIdx.x+i*iOffsetBetweenDelayValueLoopIterations_32bitWords;
        int iSharedMemoryIndex = threadIdx.x+i*iOffsetBetweenDelayValueLoopIterations_32bitWords;
        ((int32_t*)psDelayValsShared)[iSharedMemoryIndex] = ((int32_t*)psDelayVals)[iGlobalMemoryIndex];
    }

    //This sync has to be here as we have loaded data into shared memory in a 
    //way that the thread that loaded it is not the one that accesses it.
    __syncthreads();

    //***** Beamform the data *****
    /** Each sequential set of 64 threads calculates a single beam. All 1024 
     *  threads are used to calculate 16 beams - this code will need to be 
     *  modified for changing numbers of beams/antennas. 
     * 
     *  This means that each thread will be allocated a single delay value 
     *  struct.
     */
    struct delay_vals sDelayValuesLocal = psDelayValsShared[threadIdx.x];

    const int iNumTransfersIn_32BitWords = INTERNAL_TIME_SAMPLES*NR_STATIONS*COMPLEXITY*sizeof(int8_t)/sizeof(int32_t);
    const int iNumTransfersOut_32BitWords = INTERNAL_TIME_SAMPLES*NR_BEAMS*COMPLEXITY*sizeof(float)/sizeof(int32_t);
    int iChannelOffsetIn_32bitWords = NR_STATIONS*NR_SAMPLES_PER_CHANNEL*COMPLEXITY*sizeof(int8_t)/sizeof(int32_t)*blockIdx.x;
    int iChannelOffsetOut_32bitWords = NR_BEAMS*NR_SAMPLES_PER_CHANNEL*COMPLEXITY*sizeof(float)/sizeof(int32_t)*blockIdx.x;
    
    //These two lines can be commented out if prefetching is not used.
    int iGlobalMemoryIndex = iThreadIndex + iChannelOffsetIn_32bitWords;
    uint32_t u32PrefetchedAntData = ((uint32_t*)pi8AntennaData)[iGlobalMemoryIndex];

    #pragma unroll
    for (int j = 0; j < NR_SAMPLES_PER_CHANNEL/INTERNAL_TIME_SAMPLES; j++)
    {
        //***** Copy a portion of the input antenna data into shared memory *****
        // A single set of [NR_STATIONS][INTERNAL_TIME_SAMPLES] is loaded into 
        // shared mempry per iteration of this outer loop
        if(iThreadIndex < iNumTransfersIn_32BitWords){
            int iSharedMemoryIndex = threadIdx.x;
            //The next three lines implement a non-prefeteched memory read - they are commented out for now
            //int iTimeOffset_32bitWords = iNumTransfersIn_32BitWords*j;
            //iGlobalMemoryIndex = iThreadIndex + iTimeOffset_32bitWords + iChannelOffsetIn_32bitWords;
            //((uint32_t*)pi8AntennaDataInShared)[iSharedMemoryIndex] = ((uint32_t*)pi8AntennaData)[iGlobalMemoryIndex];

            /** The next six lines implement a prefetched memory read - i did not detect any performance improvements 
             *  testing this on the Nvidia 1080, but maybe it will make things faster on different cards.
             */
            ((uint32_t*)pi8AntennaDataInShared)[iSharedMemoryIndex] = u32PrefetchedAntData;
            if(j != NR_SAMPLES_PER_CHANNEL/INTERNAL_TIME_SAMPLES-1){
                int iTimeOffset_32bitWords = iNumTransfersIn_32BitWords*(j+1);
                iGlobalMemoryIndex = iThreadIndex + iTimeOffset_32bitWords + iChannelOffsetIn_32bitWords;
                uint32_t u32PrefetchedAntData = ((uint32_t*)pi8AntennaData)[iGlobalMemoryIndex];
            }
        }
        /** This __syncthreads() is here as the reading from global memory 
         *  does necessarily follow the same thread indexing convention as 
         *  generating the steering coeffs - this is done to ensure global 
         *  memory reads are properly coalesced.
         */
        __syncthreads();

        //These values are used for determining the antenna sample index 
        //relevant to the current thread
        int iBeamIndex = iThreadIndex/NR_STATIONS;
        int iAntIndex = iThreadIndex - iBeamIndex * NR_STATIONS;

        //This inner loop performs beamforing for each INTERNAL_TIME_SAMPLE
        #pragma unroll
        for (int i = 0; i < INTERNAL_TIME_SAMPLES; i++)
        {
            //***** Get Antenna Sample Values *****
            //int iSampleIndex = COMPLEXITY*(iAntIndex*INTERNAL_TIME_SAMPLES + i);
            int8_t i8AntValueReal = pi8AntennaDataInShared[iAntIndex][i][0];//Performance bottleneck 
            int8_t i8AntValueImag = pi8AntennaDataInShared[iAntIndex][i][1];//Performance bottleneck
            
            //***** Calculate Steering Coefficients *****
            int iTimeIndex = i + j * INTERNAL_TIME_SAMPLES;
            float fDeltaTime = iTimeIndex*SAMPLING_PERIOD*FFT_SIZE;
            float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
            float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
            float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2.0f)*((float)M_PI)/(SAMPLING_PERIOD*((float)NR_CHANNELS));
            float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
            float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
            float fRotation = fDelayN + fPhase0;
            float fSteeringCoeffCorrectReal;
            float fSteeringCoeffCorrectImag;
            __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
            
            //***** Multiply Antenna Sample by steering coefficient *****
            float fTempOutReal = fSteeringCoeffCorrectReal * ((float)i8AntValueReal); 
            float fTempOutImag = fSteeringCoeffCorrectImag * ((float)i8AntValueImag); 

            int iThreadIndexInWarp = iAntIndex%32;
            int iWarpIndex = iAntIndex/32;
            
            //***** Reduction operation *****
            /** Each thread produces values for a single antenna beam 
             *  combination. Per beam, every antenna needs to be summed up. A
             *  simple redcution 
             */  
            #pragma unroll
            for (int iStep = 2; iStep <= 32; iStep=iStep<<1)
            {
                int iThreadsPerWarp = 32/iStep;
                uint32_t u32WarpMask = __ballot_sync(0xffffffff, iThreadIndexInWarp < iThreadsPerWarp);
                float fTempOtherThreadReal = __shfl_down_sync(u32WarpMask,fTempOutReal,iThreadsPerWarp);
                float fTempOtherThreadImag = __shfl_down_sync(u32WarpMask,fTempOutImag,iThreadsPerWarp);
                fTempOutReal+=fTempOtherThreadReal;
                fTempOutImag+=fTempOtherThreadImag;
            }
            //Add the sums of each warp together
            if(iThreadIndexInWarp == 0)
            {
                warpSums[iBeamIndex][i][iWarpIndex][0] = fTempOutReal;
                warpSums[iBeamIndex][i][iWarpIndex][1] = fTempOutImag;
            }
        }
        /** This __syncthreads() is here as the writing back to global memory 
         *  does necessarily follow the same thread indexing convention as 
         *  generating the steering coeffs - this is done to ensure global 
         *  memory writes are properly coalesced.
         */
        __syncthreads();

        //***** Writing shared memory struct out to global memory *****
        //Summing of the warpSums values also occurs here as it seems to be 
        //faster than doing it elsewhere.
        int iTimeOffsetOut_32bitWords = iNumTransfersOut_32BitWords*j;
        if(iThreadIndex < iNumTransfersOut_32BitWords)
        {
            int iGlobalMemoryIndex = iThreadIndex + iTimeOffsetOut_32bitWords + iChannelOffsetOut_32bitWords;
            int iSharedMemBeamIndex = (iThreadIndex)/(INTERNAL_TIME_SAMPLES*COMPLEXITY);
            int iSharedMemTimeVal = (iThreadIndex - iSharedMemBeamIndex*INTERNAL_TIME_SAMPLES*COMPLEXITY)/(COMPLEXITY);
            int iSharedMemComplex = iThreadIndex % COMPLEXITY;
            //This warpsums addition is hard coded to two warps per beam. This will need to be changed when moving to multiple beams
            pfBeams[iGlobalMemoryIndex] = warpSums[iSharedMemBeamIndex][iSharedMemTimeVal][0][iSharedMemComplex] + warpSums[iSharedMemBeamIndex][iSharedMemTimeVal][1][iSharedMemComplex];
        }
    }
    
    __syncthreads();
    
}