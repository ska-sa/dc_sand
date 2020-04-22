#pragma once

#define NR_CHANNELS 64 //Number of frequency channels
//#define NR_POLARIZATIONS Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 1 //Number of time samples per channel
#define NR_STATIONS 84 //NUmber of antennas
#define NR_BEAMS 10

#define NUM_THREADS_PER_BLOCK 128

struct delay_vals_extended {
    float fSamplingPeriod_s; // seconds. Inverse of sampling frequency.
    struct timespec sRefTime_ns; //epoch time, nanosecond precision
    float fDelay_s; // seconds
    float fDelayRate_sps; //seconds per second
    float fPhase_rad; //radians
    float fPhaseRate_radps; //radians per second
};