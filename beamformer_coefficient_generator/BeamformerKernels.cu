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
    //__shared__ struct delay_vals_extended dv_shared[NUM_THREADS_PER_BLOCK];
    //int iBlockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    int iInterChannelIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(iInterChannelIndex < NR_BEAMS*NR_STATIONS){
        //Determine Correct Indices
        //int threadId= iBlockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
        int iChannelIndex = blockIdx.y; 
        int iAntIndex = iInterChannelIndex/NR_BEAMS;
        int iBeamIndex = iInterChannelIndex - iAntIndex * NR_BEAMS;

        //Calculate Values
        struct delay_vals sDelayValuesLocal = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];

        //TODO: Add logic to detect if nanoseconds go above 999999999 and increments seconds by 1
        float fTimeDifference = sCurrentTime.tv_sec - sRefTime.tv_sec;
        long fNanosecondsTimeDifference = sCurrentTime.tv_nsec - sRefTime.tv_nsec;
        fTimeDifference += (float) fNanosecondsTimeDifference / 1e9f; //Should work if this is negative as well?
        

        float fDeltaTime = fTimeDifference;
        float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
        float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
        float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
        float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
        float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
        float fRotation = fDelayN + fPhase0;
        float fSteeringCoeffCorrectReal;// = __cosf(fRotation);//At least i think its the real one - may need to check this if its important
        float fSteeringCoeffCorrectImag;// = __sinf(fRotation);

        //Fancy cuda function to combine cosf and sinf into a single operation
        __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
        
        //Write generated values to file
        size_t ulOutputIndex = (iChannelIndex*NR_STATIONS*NR_BEAMS + iInterChannelIndex)*2;
        pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;//; = //make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
        pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;

        //Useful for debuggin - leaving it here
        //if(i32OutputIndex==52016 || i32OutputIndex==52018 || i32OutputIndex==52020){
        //    printf("%i C: %i, A: %i, B: %i, r: %f, i: %f, Thread id %i, InterChannelIndex: %i, Time difference: %f\n",(channelIndex*NR_STATIONS*NR_BEAMS + interChannelIndex)*2,channelIndex,antIndex,beamIndex,fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag,threadId,interChannelIndex,time_difference);
        //}
    }

}



__global__ void calculate_beamweights_grouped_channels(
                                struct timespec sCurrentTime, 
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs,
                                bool b16BitOutput)
{
    //size_t delay_vals_length = 64*300;//n_antennas*n_beams;
    //int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    int iInterChannelIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(iInterChannelIndex < NR_BEAMS*NR_STATIONS){
        int iAntIndex = iInterChannelIndex/NR_BEAMS;
        int iBeamIndex = iInterChannelIndex - iAntIndex * NR_BEAMS;

        //Calculate Values
        struct delay_vals sDelayValuesLocal = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];

        //Calculate which channels to iterate through
        int iChannelBlockIndex = blockIdx.y;
        int iChannelStartIndex = iChannelBlockIndex*NUM_CHANNELS_PER_KERNEL;
        int iChannelStopIndex = (iChannelBlockIndex+1)*(NUM_CHANNELS_PER_KERNEL);
        if(iChannelStartIndex > NR_CHANNELS){
            iChannelStopIndex = NR_CHANNELS;
        }

        //TODO: Add logic to detect if nanoseconds go above 999999999 and increments seconds by 1
        float fTimeDifference = sCurrentTime.tv_sec - sRefTime.tv_sec;
        long fNanosecondsTimeDifference = sCurrentTime.tv_nsec - sRefTime.tv_nsec;
        fTimeDifference += (float) fNanosecondsTimeDifference / 1e9f; //Should work if this is negative as well?


        float fDeltaTime = fTimeDifference;
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
            float fSteeringCoeffCorrectReal;// = __cosf(fRotation);//At least i think its the real one - may need to check this if its important
            float fSteeringCoeffCorrectImag;// = __sinf(fRotation);
            sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
            ulOutputIndex = (iChannelIndex*NR_STATIONS*NR_BEAMS + iInterChannelIndex);
            
            if(!b16BitOutput){
                ulOutputIndex*=2;
                pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;//; = //make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
            }else{
                __half2 h2PackedOutput = __floats2half2_rn(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                //printf("Orig %f Converted %f %f\n",fSteeringCoeffCorrectReal,__high2float(h2PackedOutput), __low2float(h2PackedOutput));
                ((__half2*)pfCplxSteeringCoeffs)[ulOutputIndex] = h2PackedOutput;
            }
        }
    }
    //printf("Thread Id %i , Block Id %i, grid: x,y,z: (%i , %i, %i) thread: x,y,z: (%i, %i, %i)\n",threadId,blockId,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);

}

__global__ void calculate_beamweights_grouped_channels_and_timestamps(
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs,
                                bool b16BitOutput)
{
    //Calculate Relevant Indices
    int iBeamAntIndex = blockIdx.x*NUM_ANTBEAMS_PER_BLOCK + threadIdx.x % NUM_ANTBEAMS_PER_BLOCK;
    //int iAntIndex = iBeamAntIndex/NR_BEAMS;
    //int iBeamIndex = iBeamAntIndex - iAntIndex * NR_BEAMS;
    
    //Create shared memory arrays
    __shared__ struct delay_vals psSDelayVals[NUM_ANTBEAMS_PER_BLOCK];
    
    //Load delays into shared memory
    //if(threadIdx.x < NUM_ANTBEAMS_PER_BLOCK){
    //    psSDelayVals[threadIdx.x] = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];
    //    //((int32_t *) psSDelayVals)[threadIdx.x] = ((int32_t *)psDelayVals)[iAntIndex*NR_BEAMS + iBeamIndex];
    //}

    //More efficient to load psDelayVals at bytes per thread instead of 16 bytes per thread, this is why the array is cast to int32_t
    if(threadIdx.x<NUM_ANTBEAMS_PER_BLOCK*4){
        //psSDelayVals[threadIdx.x] = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];
        int iStartOffset =  blockIdx.x*NUM_ANTBEAMS_PER_BLOCK*4;
        ((int32_t *) psSDelayVals)[threadIdx.x] = ((int32_t *)psDelayVals)[(iStartOffset + threadIdx.x)];
    }

    __syncthreads();
    //Calculate Steering Coefficients
    if(iBeamAntIndex < NR_BEAMS*NR_STATIONS){
        struct delay_vals sDelayValuesLocal = psSDelayVals[threadIdx.x % NUM_ANTBEAMS_PER_BLOCK];
        //printf("%f %f %f %f\n",sDelayValuesLocal.fDelay_s,sDelayValuesLocal.fDelayRate_sps,sDelayValuesLocal.fPhase_rad,sDelayValuesLocal.fPhaseRate_radps);
        //Each block is divided into seperate discrete chunks. It may have made sense to have these as a y dimension instead. I may get to that later
        const int iTotalTimeChunksPerKernel = NUM_THREADS_PER_BLOCK_MAX/(NUM_ANTBEAMS_PER_BLOCK*2);//This multiply by two is here to increase the width of a single time chunk. This will better coalesce the writing of data to the pfCplxSteeringCoeffs. After some tweaking, 2 works best for 32-bit floating point outputs and 1 for 16-bit floating point outputs. When this is changed, the multiply by two in iTimeChunk two lines down must also be changed 
        const int iTimeIterations = NR_SAMPLES_PER_CHANNEL/iTotalTimeChunksPerKernel;
        int iTimeChunk = threadIdx.x / (NUM_ANTBEAMS_PER_BLOCK*2);
        //#pragma unroll
        for (int iTimeStep = 0; iTimeStep < iTimeIterations; iTimeStep++)
        {
            int iTimeIndex = iTimeChunk + iTimeStep*iTotalTimeChunksPerKernel;
            float fTimeDifference = iTimeIndex*SAMPLING_PERIOD*FFT_SIZE; //Should work if this is negative as well?
            float fDeltaTime = fTimeDifference;
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
                float fSteeringCoeffCorrectReal;// = __cosf(fRotation);//At least i think its the real one - may need to check this if its important
                float fSteeringCoeffCorrectImag;// = __sinf(fRotation);
                __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);

                if(!b16BitOutput){
                    //32 bit output
                    ulOutputIndex = (NR_STATIONS*NR_CHANNELS*NR_BEAMS*iTimeIndex + iChannelIndex*NR_STATIONS*NR_BEAMS + iBeamAntIndex)*2;
                    pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;//; = //make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                    pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
                }else{
                    ulOutputIndex = (NR_STATIONS*NR_CHANNELS*NR_BEAMS*iTimeIndex + iChannelIndex*NR_STATIONS*NR_BEAMS + iBeamAntIndex);
                    //float2 f2Temp;
                    //f2Temp.x = fSteeringCoeffCorrectReal;
                    //f2Temp.y = fSteeringCoeffCorrectImag;
                    __half2 h2PackedOutput = __floats2half2_rn(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
                    //printf("Orig %f Converted %f %f\n",fSteeringCoeffCorrectReal,__high2float(h2PackedOutput), __low2float(h2PackedOutput));
                    ((__half2*)pfCplxSteeringCoeffs)[ulOutputIndex] = h2PackedOutput;
                }
            }
        }
    }
    //Store steering coefficients into shared memory
}

//TODO Iterate over a number of channels to reduce the shared memory usage
__global__ void calculate_beamweights_and_beamform_single_channel(
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfBeams,
                                int8_t * pi8AntennaData)
{
    __shared__ delay_vals psDelayValsShared[NR_STATIONS*NR_BEAMS];
    //__shared__ float pfBeamReductionStore[NR_BEAMS][NR_STATIONS][COMPLEXITY];
    __shared__ int8_t pi8AntennaDataInShared[NR_STATIONS][INTERNAL_TIME_SAMPLES][COMPLEXITY];
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
        //printf("%f\n",((float*)psDelayValsShared)[iGlobalMemoryIndex]);
    }

    __syncthreads();

    //***** Beamform the data *****
    //Each sequential set of 64 threads calculates a single beam. All 1024 threads are used this way - needs to be modified for changing number of antennas
    //Can eventually take advantage of warp level operations

    //__shared__ float pfBeamDataOutShared[NR_BEAMS][INTERNAL_TIME_SAMPLES][COMPLEXITY];
    struct delay_vals sDelayValuesLocal = psDelayValsShared[threadIdx.x];

    // if(iThreadIndex == 0 && iChannelIndex == 0){
    //     printf("%f %f %f %f\n",psDelayValsShared[threadIdx.x].fDelay_s,sDelayValuesLocal.fDelayRate_sps,sDelayValuesLocal.fPhase_rad,sDelayValuesLocal.fPhaseRate_radps);
    // }

    const int iNumTransfersIn_32BitWords = INTERNAL_TIME_SAMPLES*NR_STATIONS*COMPLEXITY*sizeof(int8_t)/sizeof(int32_t);
    const int iNumTransfersOut_32BitWords = INTERNAL_TIME_SAMPLES*NR_BEAMS*COMPLEXITY*sizeof(float)/sizeof(int32_t);
    int iChannelOffsetIn_32bitWords = NR_STATIONS*NR_SAMPLES_PER_CHANNEL*COMPLEXITY*sizeof(int8_t)/sizeof(int32_t)*blockIdx.x;
    int iChannelOffsetOut_32bitWords = NR_BEAMS*NR_SAMPLES_PER_CHANNEL*COMPLEXITY*sizeof(float)/sizeof(int32_t)*blockIdx.x;
    
    //***** Perform Beamforming *****
    #pragma unroll
    for (int j = 0; j < NR_SAMPLES_PER_CHANNEL/INTERNAL_TIME_SAMPLES; j++)
    {
        int iTimeOffset_32bitWords = iNumTransfersIn_32BitWords*j;
        //***** Copy a portion of the input antenna data into shared memory *****
        if(iThreadIndex < iNumTransfersIn_32BitWords){
            int iGlobalMemoryIndex = iThreadIndex + iTimeOffset_32bitWords + iChannelOffsetIn_32bitWords;
            int iSharedMemoryIndex = threadIdx.x;
            ((uint32_t*)pi8AntennaDataInShared)[iSharedMemoryIndex] = ((uint32_t*)pi8AntennaData)[iGlobalMemoryIndex];
            //((uint32_t *)pfBeams)[iGlobalMemoryIndex] = ((uint32_t*)pi8AntennaDataInShared)[iSharedMemoryIndex];
            //if(j == 0 && iChannelIndex == 0){
            //    printf("ThreadID: %d, GlobMemIndex: %d, Input: %d, Output: %d\n",iThreadIndex,iGlobalMemoryIndex,((uint32_t*)pi8AntennaData)[iThreadIndex + iTimeOffset_32bitWords + iChannelOffset_32bitWords],((uint32_t *)pfBeams)[iThreadIndex + iTimeOffset_32bitWords + iChannelOffset_32bitWords]);
            //}
        }
        __syncthreads();

        //These values are used for determining the antenna sample index relevant to the current thread
        int iBeamIndex = iThreadIndex/NR_STATIONS;
        int iAntIndex = iThreadIndex - iBeamIndex * NR_STATIONS;

        //These 8 indeces are for the reduction operation, but they are declared here as they dont need to be part of the for loop
        //int iIndex0In = 2*iAntIndex + 0;// + iBeamIndex*NR_STATIONS*COMPLEXITY;
        //int iImagIndex0In = 4*iAntIndex + 1;// + iBeamIndex*NR_STATIONS*COMPLEXITY;
        //int iIndex1In = 2*iAntIndex + 1;// + iBeamIndex*NR_STATIONS*COMPLEXITY;
        //int iImagIndex1In = 4*iAntIndex + 3;// + iBeamIndex*NR_STATIONS*COMPLEXITY;
        //int iIndexOut = iAntIndex + 0;// + iBeamIndex*NR_STATIONS*COMPLEXITY;
        //int iImagIndexOut = 2*iAntIndex + 1;// + iBeamIndex*NR_STATIONS*COMPLEXITY;

        #pragma unroll
        for (int i = 0; i < INTERNAL_TIME_SAMPLES; i++)
        {
            //***** Get Antenna Sample Values *****
            //int iSampleIndex = COMPLEXITY*(iAntIndex*INTERNAL_TIME_SAMPLES + i);
            int8_t i8AntValueReal = pi8AntennaDataInShared[iAntIndex][i][0];//Performance bottleneck 
            int8_t i8AntValueImag = pi8AntennaDataInShared[iAntIndex][i][1];//Performance bottleneck
            
            //***** Calculate Steering Coefficients *****
            int iTimeIndex = i + j * INTERNAL_TIME_SAMPLES;
            float fTimeDifference = iTimeIndex*SAMPLING_PERIOD*FFT_SIZE; //Should work if this is negative as well?
            float fDeltaTime = fTimeDifference;
            float fDeltaDelay = sDelayValuesLocal.fDelayRate_sps*fDeltaTime;
            float fDeltaPhase = sDelayValuesLocal.fPhaseRate_radps*fDeltaTime;
            float fDelayN2 = (sDelayValuesLocal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2.0f)*((float)M_PI)/(SAMPLING_PERIOD*((float)NR_CHANNELS));
            float fDelayN = (sDelayValuesLocal.fDelayRate_sps + fDeltaDelay)*iChannelIndex*((float)M_PI)/(SAMPLING_PERIOD*NR_CHANNELS);
            float fPhase0 = sDelayValuesLocal.fPhase_rad - fDelayN2 + fDeltaPhase;
            float fRotation = fDelayN + fPhase0;
            float fSteeringCoeffCorrectReal;// = __cosf(fRotation);//At least i think its the real one - may need to check this if its important
            float fSteeringCoeffCorrectImag;// = __sinf(fRotation);
            __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);

            //***** Multiply Antenna Sample by steering coefficient *****
            //pfBeamReductionStore[iBeamIndex][iAntIndex][0] = iAntIndex*COMPLEXITY;//fSteeringCoeffCorrectReal * ((float)i8AntValueReal);//Performance bottleneck goes from 0.2ms to 0.6ms kernel run time - need to figure it out
            //pfBeamReductionStore[iBeamIndex][iAntIndex][1] = iAntIndex*COMPLEXITY + 1;//fSteeringCoeffCorrectImag * ((float)i8AntValueImag);//Performance bottleneck

            // if(iChannelIndex == 0 && iThreadIndex < 16 && j == 1 && i==0){
            //     __syncthreads();
            //     printf("%f Ant: %d CoeffIndex: %d Real: %f Imag: %f, Ant Val real: %d, Ant Val Imag: %d\n\t Delay Vals: %f %f %f %f\n\tSample*BW Real %f Imag %f\n",fRotation ,iAntIndex, iThreadIndex, fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag,(int32_t)i8AntValueReal,(int32_t)i8AntValueImag,sDelayValuesLocal.fDelay_s*1000000,sDelayValuesLocal.fDelayRate_sps*1000000,sDelayValuesLocal.fPhase_rad*1000000,sDelayValuesLocal.fPhaseRate_radps*1000000,pfBeamReductionStore[iThreadIndex*COMPLEXITY],pfBeamReductionStore[iThreadIndex*COMPLEXITY+1]);
            //      //printf("%d %f %f %f %f\nSteering Coeffs. Ant: %d Real %f Imag %f\n",threadIdx.x,sDelayValuesLocal.fDelay_s*1000000,sDelayValuesLocal.fDelayRate_sps*1000000,sDelayValuesLocal.fPhase_rad*1000000,sDelayValuesLocal.fPhaseRate_radps*1000000,iThreadIndex,fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
            // }


            //__syncthreads();
            //***** Sum Coefficient Values Together - Adds real to real - basic reduce algorithm, to be replaced with cuda warp level primitives at a later date ****
            //The shared memory writes  pfBeamReductionStore[iRealIndexOut] = iTempOutReal are expensive, I will try implement this with warp level primitives 
            //This section is a bit of a bottleneck - runtime goes from 0.75 ms to 1.03ms

            float fTempOutReal = fSteeringCoeffCorrectReal * ((float)i8AntValueReal); //pfBeamReductionStore[iBeamIndex][iIndex0In][0];
            float fTempOutImag = fSteeringCoeffCorrectImag * ((float)i8AntValueImag); //pfBeamReductionStore[iBeamIndex][iIndex0In][1];

            int iThreadIndexInWarp = iAntIndex%32;
            int iWarpIndex = iAntIndex/32;
            

            // if(i == 0 && iChannelIndex == 0 && iAntIndex < 64 && j==0 && iBeamIndex == 0)
            // {
            //     printf("A: %d %f %f\n",iAntIndex,fTempOutReal,fTempOutImag);
            // }
            //__syncthreads();

            #pragma unroll
            for (int iStep = 2; iStep <= 32; iStep=iStep<<1)//Warp wise thread
            {
                int iThreadsPerWarp = 32/iStep;
                uint32_t u32WarpMask = __ballot_sync(0xffffffff, iThreadIndexInWarp < iThreadsPerWarp);
                float fTempOtherThreadReal = __shfl_down_sync(u32WarpMask,fTempOutReal,iThreadsPerWarp);
                float fTempOtherThreadImag = __shfl_down_sync(u32WarpMask,fTempOutImag,iThreadsPerWarp);
                // __syncthreads();
                // if(i == 0 && iChannelIndex == 0 && iAntIndex < 64 && j==0 && iBeamIndex == 0 && iThreadIndexInWarp < iThreadsPerWarp)
                // {
                //     __syncthreads();
                //     printf("Warp Index: % d B: Ant: %d This Thread: %f, Other Thread(%d):  %f Warp Mask 0x%x\n",iThreadIndexInWarp,iAntIndex,fTempOutReal,iThreadIndexInWarp+32/iStep,fTempOtherThreadReal,u32WarpMask);
                // }
                // __syncthreads();
                // if(i == 0 && iChannelIndex == 0 && iAntIndex < 64 && j==0 && iBeamIndex == 0 && iAntIndex == 0)
                // {
                //     printf("\n");
                // }
                fTempOutReal+=fTempOtherThreadReal;
                fTempOutImag+=fTempOtherThreadImag;
            }
            //Add the sums of each warp together
            if(iThreadIndexInWarp == 0){
                warpSums[iBeamIndex][i][iWarpIndex][0] = fTempOutReal;
                warpSums[iBeamIndex][i][iWarpIndex][1] = fTempOutImag;
            }
            //__syncthreads();
            //if(iAntIndex == 0){
            //   pfBeamDataOutShared[iBeamIndex][i][0] = fTempOutReal + warpSums[iBeamIndex][i][1][0];//Hardcoded for two warps, needs to be fixed
            //   pfBeamDataOutShared[iBeamIndex][i][1] = fTempOutImag + warpSums[iBeamIndex][i][1][1];
            //}

            //**** Writing formed beam value to shared memory struct *****
            //packed as [NR_BEAMS][INTENRAL_TIME_SAMPLES] - this could be improved, but no discussions around that have happened yet
            //int iSharedIndexOffset = 2*(iBeamIndex*INTERNAL_TIME_SAMPLES + i);
            // if(iAntIndex == 0){
            //     pfBeamDataOutShared[iBeamIndex][i][0] = pfBeamReductionStore[iBeamIndex][iIndexOut][0];
            //     pfBeamDataOutShared[iBeamIndex][i][1] = pfBeamReductionStore[iBeamIndex][iIndexOut][1];
            //     //if(iChannelIndex == 0 && iThreadIndex == 0 && i == 0 && j == 0){
            //     //    __syncthreads();
            //     //    printf("\nThread Data: %d %d %d %d %f %f\n",j,i,iRealIndexOut,iImagIndexOut,pfBeamReductionStore[iRealIndexOut],pfBeamReductionStore[iImagIndexOut]);
            //     //    //printf("%d %d %d %f %f\n",j,i,iSharedIndexOffset,pfBeamDataOutShared[iSharedIndexOffset],pfBeamDataOutShared[iSharedIndexOffset+1]);
            //     //}
            // }
        }
        __syncthreads();//This is here as the writing back to global memory does not necessarily follow the same thread indexing convention as generating the steering coeffs - dont hate me
        
        // if(iChannelIndex == 0 && iThreadIndex == 0 && j == 1){
        //     __syncthreads();
        //     printf("\nShared Memory Beams %d %f %f\n",j,pfBeamDataOutShared[0],pfBeamDataOutShared[1]);
        // }

        //***** Writing shared memory struct out to global memory *****
        int iTimeOffsetOut_32bitWords = iNumTransfersOut_32BitWords*j;
        if(iThreadIndex < iNumTransfersOut_32BitWords){
            int iGlobalMemoryIndex = iThreadIndex + iTimeOffsetOut_32bitWords + iChannelOffsetOut_32bitWords;
            //int iSharedMemoryIndex = threadIdx.x;
            int iSharedMemBeamIndex = (iThreadIndex)/(INTERNAL_TIME_SAMPLES*COMPLEXITY);
            int iSharedMemTimeVal = (iThreadIndex-iSharedMemBeamIndex*INTERNAL_TIME_SAMPLES*COMPLEXITY)/(COMPLEXITY);
            int iSharedMemComplex = iThreadIndex%COMPLEXITY;
            //((uint32_t*)pi8AntennaDataInShared)[iSharedMemoryIndex] = ((uint32_t*)pi8AntennaData)[iGlobalMemoryIndex];
            pfBeams[iGlobalMemoryIndex] = warpSums[iSharedMemBeamIndex][iSharedMemTimeVal][0][iSharedMemComplex] + warpSums[iSharedMemBeamIndex][iSharedMemTimeVal][1][iSharedMemComplex];
            // if(j == 1 && iChannelIndex == 0 && iThreadIndex == 0){
            //     __syncthreads();
            //     //printf("ThreadID: %d, GlobMemIndex: %d, Input: %f, Output: %f\n",iThreadIndex,iGlobalMemoryIndex,pfBeamDataOutShared[iSharedMemoryIndex],pfBeams[iGlobalMemoryIndex]);
            //     printf("%d %d %d %d %f %f\n",iThreadIndex,iGlobalMemoryIndex,iTimeOffsetOut_32bitWords,iChannelOffsetOut_32bitWords,pfBeamDataOutShared[iSharedMemoryIndex],pfBeams[iGlobalMemoryIndex]);
            // }
        }

        // if(iChannelIndex == 0 && iThreadIndex == 0 && j == 0){
        //     __syncthreads();
        //     printf("\nGlobal Memory %d %f %f\n",j,pfBeams[0],pfBeams[1]);
        // }
    }
    
    __syncthreads();
    
}