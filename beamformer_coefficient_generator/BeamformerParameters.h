#pragma once

#define NR_CHANNELS 64 //Number of frequency channels
//#define NR_POLARIZATIONS Just one for now, but I would eventually like to generalise it.
#define NR_SAMPLES_PER_CHANNEL 1 //Number of time samples per channel
#define NR_STATIONS 84 //NUmber of antennas
#define NR_BEAMS 10

#define NUM_THREADS_PER_BLOCK 128