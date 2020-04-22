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
        struct delay_vals_extended my_dv = dv[antIndex*NR_BEAMS + beamIndex];
        //struct delay_vals my_dv = dv[antIndex*n_beams + beamIndex];
        //1

        float time_difference = (float) current_time.tv_sec - (float) my_dv.sRefTime_ns.tv_sec;
        long nanosec_difference = current_time.tv_nsec - my_dv.sRefTime_ns.tv_nsec;
        time_difference += (float) nanosec_difference / 1e9; //Should work if this is negative as well?
        

        float delta_time = time_difference;//ts_diff(my_dv.ref_time, current_time);
        float delta_delay = my_dv.fDelayRate_sps*delta_time;
        //2
        float delay_n = (my_dv.fDelay_s + delta_delay)*channelIndex*M_PI/(my_dv.fSamplingPeriod_s*NR_CHANNELS);
        //3
        float delay_N_2 = (my_dv.fDelay_s + delta_delay)*(NR_CHANNELS/2)*M_PI/(my_dv.fSamplingPeriod_s*NR_CHANNELS);
        //4
        float delta_phase = my_dv.fPhaseRate_radps*delta_time;
        // //5
        float phase_0 = my_dv.fPhase_rad - delay_N_2 + delta_phase;
        // //6
        float rotation = delay_n + phase_0;
        //printf("%f\n",rotation);
        ((cuFloatComplex*)cplx_beamweights)[(channelIndex*NR_STATIONS*NR_BEAMS + interChannelIndex)] = make_cuFloatComplex(cosf(rotation),sinf(rotation));
        //printf("C: %i, A: %i, B: %i, r: %f, i: %f, Thread id %i, InterChannelIndex: %i\n",channelIndex,antIndex,beamIndex,cos(rotation),sin(rotation),threadId,interChannelIndex);
    }
    //printf("Thread Id %i , Block Id %i, grid: x,y,z: (%i , %i, %i) thread: x,y,z: (%i, %i, %i)\n",threadId,blockId,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);
}