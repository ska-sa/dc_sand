#ifndef __BEAMFORMER_PARAMETERS_H__
#define __BEAMFORMER_PARAMETERS_H__

#define NR_CHANNELS 64 //Number of frequency channels
#define NR_POLARIZATIONS 1 // Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 256 //Number of time samples per channel. Must be a multiple of 64
#define NR_STATIONS 64 //NUmber of antennas
#define NR_BEAMS 16

//Values used to calculate delays
#define SAMPLING_PERIOD 1e-9f
#define FFT_SIZE 8192
#define ADC_SAMPLE_RATE 1712e6
#define ACCUMULATIONS_BEFORE_NEW_COEFFS 256

/** Limit the number of threads per block in the naive kernel - this prevents too
 *  many threads being wasted when the number of used threads is significantly 
 *  less than the maximum of 1024 threads per block.
 */ 
#define NUM_THREADS_PER_BLOCK 128

//Used for the calculate_beamweights_grouped_channels kernel
#define NUM_CHANNELS_PER_KERNEL NR_CHANNELS

//Used for the calculate_beamweights_grouped_channels_and_timestamps kernel
#define NUM_ANTBEAMS_PER_BLOCK 16 //Must be a power of 2
#define NUM_THREADS_PER_BLOCK_MAX 1024

/** \brief Delay values struct that stores the values transmitted from the SARAO CAM team.
 * 
 *  \details Delay values struct that stores the values transmitted from the SARAO CAM team.
 *           There is one of these for every antenna beam combinations for NR_STATIONS*NR_BEAMS of these structs
 */
struct delay_vals{
    float fDelay_s; // seconds
    float fDelayRate_sps; //seconds per second
    float fPhase_rad; //radians
    float fPhaseRate_radps; //radians per second
};

#endif