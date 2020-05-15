#ifndef __BEAMFORMER_PARAMETERS_H__
#define __BEAMFORMER_PARAMETERS_H__

#define COMPLEXITY 2 //Used to improve readability

//These parameters shape the 
#define NR_CHANNELS 64 //512 //Number of frequency channels
#define NR_POLARIZATIONS 1 // Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 256 //Number of time samples per channel. Must be a multiple of 64
#define NR_STATIONS 64 //NUmber of antennas
#define NR_BEAMS 16

//Values used to calculate delays
#define SAMPLING_PERIOD 1e-7f
#define FFT_SIZE 8192
#define ADC_SAMPLE_RATE 1712e6
#define ACCUMULATIONS_BEFORE_NEW_COEFFS 256

/** Limit the number of threads per block in the naive kernel - this prevents 
 *  too many threads being wasted when the number of used threads is 
 *  significantly less than the maximum of 1024 threads per block.
 */ 
#define NUM_THREADS_PER_BLOCK 128

/** Used for the calculate_beamweights_grouped_channels kernel - at the moment 
 * it is equal to the total number of channels, but future versions may make each
 * kernel work on a subset of the channels
 */
#define NUM_CHANNELS_PER_KERNEL NR_CHANNELS

//Used for the calculate_beamweights_grouped_channels_and_timestamps kernel
/**
 * For the calculate_beamweights_grouped_channels_and_timestamps kernel, each 
 * block only operates on a subset of 
 */
#define NUM_ANTBEAMS_PER_BLOCK 16 //Must be a power of 2

/**
 * This is standard for CUDA, this define just makes things more explicit.
 */ 
#define NUM_THREADS_PER_BLOCK_MAX 1024

//Used for calculate_beamweights_and_beamform_single_channel kernel
/**
 * The ASTRON Tensor core correlation kernels have an interesting input data 
 * format requirements of char2 [channels][time/16][station][16]. The time index
 * is split into two samples. The INTERNAL_TIME_SAMPLES makes this explicit in 
 * the code
 */ 
#define INTERNAL_TIME_SAMPLES 16 

/** \brief  Delay values struct that stores the values transmitted from the 
 *          SARAO CAM team.
 * 
 *  \details    Delay values struct that stores the values transmitted from the 
 *              SARAO CAM team. There is one of these for every antenna beam 
 *              combinations for a total of NR_STATIONS*NR_BEAMS of these 
 *              structs
 */
struct delay_vals{
    float fDelay_s; // seconds
    float fDelayRate_sps; //seconds per second
    float fPhase_rad; //radians
    float fPhaseRate_radps; //radians per second
};

#endif