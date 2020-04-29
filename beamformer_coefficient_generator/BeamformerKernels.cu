#include <stdio.h> 
#include "BeamformerParameters.h"
#include "cuComplex.h"
#include <cstdint>

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
                                float* pfCplxSteeringCoeffs)
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
            __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);
            ulOutputIndex = (iChannelIndex*NR_STATIONS*NR_BEAMS + iInterChannelIndex)*2;
            pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;//; = //make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
            pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
        }
    }
    //printf("Thread Id %i , Block Id %i, grid: x,y,z: (%i , %i, %i) thread: x,y,z: (%i, %i, %i)\n",threadId,blockId,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);

}

__global__ void calculate_beamweights_grouped_channels_and_timestamps(
                                struct timespec sRefTime,
                                struct delay_vals *psDelayVals, 
                                float* pfCplxSteeringCoeffs)
{
    //Calculate Relevant Indices
    int iBeamAntIndex = blockIdx.x*NUM_ANTBEAMS_PER_BLOCK + threadIdx.x % NUM_ANTBEAMS_PER_BLOCK;
    int iAntIndex = iBeamAntIndex/NR_BEAMS;
    int iBeamIndex = iBeamAntIndex - iAntIndex * NR_BEAMS;
    int iTimeIndex = threadIdx.x / NUM_ANTBEAMS_PER_BLOCK;

    //Create shared memory arrays
    __shared__ struct delay_vals psSDelayVals[NUM_ANTBEAMS_PER_BLOCK];

    float fTimeDifference = iTimeIndex*SAMPLING_PERIOD*FFT_SIZE; //Should work if this is negative as well?
    
    //Load delays into shared memory
    if(iTimeIndex==0){
        psSDelayVals[threadIdx.x] = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];
        //((int32_t *) psSDelayVals)[threadIdx.x] = ((int32_t *)psDelayVals)[iAntIndex*NR_BEAMS + iBeamIndex];
    }

    //if(iTimeIndex<4){
        //psSDelayVals[threadIdx.x] = psDelayVals[iAntIndex*NR_BEAMS + iBeamIndex];
    //    int iStartOffset =  blockIdx.x*NUM_ANTBEAMS_PER_BLOCK*4;
        //printf("%d\n",iStartOffset);
    //    ((int32_t *) psSDelayVals)[threadIdx.x] = ((int32_t *)psDelayVals)[(iStartOffset + threadIdx.x)];
    //}

    __syncthreads();
    //Calculate Steering Coefficients
    if(iBeamAntIndex < NR_BEAMS*NR_STATIONS){
        struct delay_vals sDelayValuesLocal = psSDelayVals[threadIdx.x % NUM_ANTBEAMS_PER_BLOCK];

        //Calculate which channels to iterate through
        int iChannelBlockIndex = 0;
        int iChannelStartIndex = iChannelBlockIndex*NUM_CHANNELS_PER_KERNEL;
        int iChannelStopIndex = (iChannelBlockIndex+1)*(NUM_CHANNELS_PER_KERNEL);
        if(iChannelStartIndex > NR_CHANNELS){
            iChannelStopIndex = NR_CHANNELS;
        }

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
            __sincosf(fRotation,&fSteeringCoeffCorrectImag,&fSteeringCoeffCorrectReal);

            ulOutputIndex = (NR_STATIONS*NR_CHANNELS*NR_BEAMS*iTimeIndex + iChannelIndex*NR_STATIONS*NR_BEAMS + iBeamAntIndex)*2;
            //if(ulOutputIndex > 10485760-50){
            //    //printf("Ya here yet?\n");
            //    printf("%li B %d A %d T %d C %d\n",ulOutputIndex,iBeamIndex,iAntIndex,iTimeIndex,iChannelIndex);
            //}
            pfCplxSteeringCoeffs[ulOutputIndex] = fSteeringCoeffCorrectReal;//; = //make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
            pfCplxSteeringCoeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag;
        }
    }
    //Store steering coefficients into shared memory
}