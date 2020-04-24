#ifndef __BEAMFORMER_PARAMETERS_H__
#define __BEAMFORMER_PARAMETERS_H__

#define NR_CHANNELS 50 //Number of frequency channels
//#define NR_POLARIZATIONS Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 100 //Number of time samples per channel
#define NR_STATIONS 90 //NUmber of antennas
#define NR_BEAMS 100

//Values used to calculate delays
#define SAMPLING_PERIOD 1e-9f
#define FFT_SIZE 8192

/** Limit the number of threads per block in the naive kernel - this prevents too
 *  many threads being wasted when the number of used threads is significantly 
 *  less than the maximum of 1024 threads per block.
 */ 
#define NUM_THREADS_PER_BLOCK 128

/** Delay values struct that stores the values transmitted from the SARAO CAM team.
 * There is one of these for every antenna beam combinations for NR_STATIONS*NR_BEAMS of these structs
 */
struct delay_vals{
    float fDelay_s; // seconds
    float fDelayRate_sps; //seconds per second
    float fPhase_rad; //radians
    float fPhaseRate_radps; //radians per second
};

#endif