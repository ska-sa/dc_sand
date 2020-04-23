#pragma once

#define NR_CHANNELS 64 //Number of frequency channels
//#define NR_POLARIZATIONS Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 64 //Number of time samples per channel
#define NR_STATIONS 84 //NUmber of antennas
#define NR_BEAMS 100

//Values used to calculate delays
#define SAMPLING_PERIOD 1e-9f
#define FFT_SIZE 8192

#define NUM_THREADS_PER_BLOCK 128


struct delay_vals{
    float fDelay_s; // seconds
    float fDelayRate_sps; //seconds per second
    float fPhase_rad; //radians
    float fPhaseRate_radps; //radians per second
};
