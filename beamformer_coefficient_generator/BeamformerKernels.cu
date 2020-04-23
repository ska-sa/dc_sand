#include <stdio.h> 
#include "BeamformerParameters.h"
#include "cuComplex.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

__global__ void calculate_beamweights_naive(
                                struct timespec current_time, 
                                struct delay_vals_extended *dv, 
                                float* cplx_beamweights)
{
    //size_t delay_vals_length = 64*300;//n_antennas*n_beams;
    //__shared__ struct delay_vals_extended dv_shared[NUM_THREADS_PER_BLOCK];
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    int interChannelIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(interChannelIndex < NR_BEAMS*NR_STATIONS){
        //Determine Correct Indices
        int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
        int channelIndex = blockIdx.y;
        int antIndex = interChannelIndex/NR_BEAMS;
        int beamIndex = interChannelIndex - antIndex * NR_BEAMS;// interChannelIndex/n_antennas;

        //Calculate Values
        //dv_shared[threadIdx.x] = dv[antIndex*NR_BEAMS + beamIndex];
        struct delay_vals_extended sDelayVal = dv[antIndex*NR_BEAMS + beamIndex];
        //struct delay_vals my_dv = dv[antIndex*n_beams + beamIndex];
        //1

        float time_difference = (float) current_time.tv_sec - (float) sDelayVal.sRefTime_ns.tv_sec;
        long nanosec_difference = current_time.tv_nsec - sDelayVal.sRefTime_ns.tv_nsec;
        time_difference += (float) nanosec_difference / 1e9; //Should work if this is negative as well?
        

        float fDeltaTime = time_difference;
        float fDeltaDelay = sDelayVal.fDelayRate_sps*fDeltaTime;
        float fDelayN = (sDelayVal.fDelayRate_sps + fDeltaDelay)*channelIndex*M_PI/(sDelayVal.fSamplingPeriod_s*NR_CHANNELS);
        float fDelayN2 = (sDelayVal.fDelay_s + fDeltaDelay)*(NR_CHANNELS/2)*M_PI/(sDelayVal.fSamplingPeriod_s*NR_CHANNELS);
        float fDeltaPhase = sDelayVal.fPhaseRate_radps*fDeltaTime;
        float fPhase0 = sDelayVal.fPhase_rad - fDelayN2 + fDeltaPhase;
        float fRotation = fDelayN + fPhase0;
        float fSteeringCoeffCorrectReal = cos(fRotation);//At least i think its the real one - may need to check this if its important
        float fSteeringCoeffCorrectImag = sin(fRotation);

        //printf("%f\n",rotation);
        ((cuFloatComplex*)cplx_beamweights)[(channelIndex*NR_STATIONS*NR_BEAMS + interChannelIndex)] = make_cuFloatComplex(fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag);
        //if((channelIndex*NR_STATIONS*NR_BEAMS + interChannelIndex)*2==1680){
        //    printf("%i C: %i, A: %i, B: %i, r: %f, i: %f, Thread id %i, InterChannelIndex: %i\n",(channelIndex*NR_STATIONS*NR_BEAMS + interChannelIndex)*2,channelIndex,antIndex,beamIndex,fSteeringCoeffCorrectReal,fSteeringCoeffCorrectImag,threadId,interChannelIndex);
        //}
        //printf("C: %i, A: %i, B: %i, r: %f, i: %f, Thread id %i, InterChannelIndex: %i\n",channelIndex,antIndex,beamIndex,cos(rotation),sin(rotation),threadId,interChannelIndex);
    }
    //printf("Thread Id %i , Block Id %i, grid: x,y,z: (%i , %i, %i) thread: x,y,z: (%i, %i, %i)\n",threadId,blockId,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);
}