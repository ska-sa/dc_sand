__global__ void precise_beamformer(float4 DelayVals[N_BEAMS][N_ANTENNAS],
                                   char2 AntennaData[N_ANTENNAS][N_CHANS][TOTAL_TIME],
                                   cuComplex Output[TOTAL_TIME][N_CHANS][N_BEAMS] )
{
    //Shared memory space to store antenna-data (AD)
    __shared__ char2 sAntennaData[TIME_STEP][N_ANTENNAS][N_CHANS]; //__attribute((aligned(32))); //not sure what this actually does?
    //Retrieve AD from global memory.
    sAntennaData
}


