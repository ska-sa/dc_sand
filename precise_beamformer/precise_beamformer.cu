__global__ void precise_beamformer(float4 DelayVals[N_BEAMS][N_ANTENNAS],
                                   char2 AntennaData[N_ANTENNAS][N_CHANS][TOTAL_TIME],
                                   cuComplex Output[TOTAL_TIME][N_CHANS][N_BEAMS] )
{
    
}