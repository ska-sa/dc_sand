#include <cuComplex.h>

#define N 8192*8
#define Fs 1712e6
#define Fir_length 256

__device__ float mixed_data_re[Fir_length + N];
__device__ float mixed_data_im[Fir_length + N];  

__global__ void kernel_ddc(float *data_in, float *fir_coeffs, float *data_downsampled_out, float osc_frequency, float *debug_data_real, float *debug_data_imag)
{
    __shared__ float fir_coeffs_shared[Fir_length];
    // __shared__ float prev_data_slice_re[Fir_length];
    // __shared__ float prev_data_slice_im[Fir_length];
    // __shared__ float mixed_data_re[256+256];
    // __shared__ float mixed_data_im[256+256];

    //Stage 1: Load Data From memory, mix and store in shared memory
    //1.1 Load FIR coeffs into shared memory
    fir_coeffs_shared[threadIdx.x] = fir_coeffs[threadIdx.x];

    // printf("Lookup state is %f\n", lookup_state);

    if (blockIdx.x == 0){
        // 1.2 Copy the trailing data (of Fir_length) to the beginning of the new data word that has been received.
        mixed_data_re[Fir_length - threadIdx.x] = mixed_data_re[Fir_length + N - threadIdx.x];
        mixed_data_im[Fir_length - threadIdx.x] = mixed_data_im[Fir_length + N - threadIdx.x];
        // printf("Blockx.x is %d and Threadx.x is %d so dest %d from %d\n", blockIdx.x, threadIdx.x, (Fir_length - threadIdx.x),(Fir_length + N - threadIdx.x));
        // debug_data_real[Fir_length - threadIdx.x] = mixed_data_re[Fir_length + N - threadIdx.x];
        // debug_data_imag[Fir_length - threadIdx.x] = mixed_data_im[Fir_length + N - threadIdx.x];
    }
    __syncthreads();

    // int inOffset = (blockIdx.x+1)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values
    int inOffset = (blockIdx.x)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values

    for(int i = 0; i <= 15; i++){ //The minus 1 accounts for the additional FIR length worth of data that we need to load of past values
    // for(int i = -1; i < 16; i++){ //The minus 1 accounts for the additional FIR length worth of data that we need to load of past values
        // 1.2 Load Data From Global Memory

        int index_in = inOffset + threadIdx.x + i*Fir_length;
        float sample_in = data_in[index_in];

        // printf("Test\n");

        // if (index_in > 16380){
        //     printf("%d %d %d %d\n", blockIdx.x,threadIdx.x,i,index_in);
        // }

        //1.3 Mix value down
        float mixerValue_re;
        float mixerValue_im;
        float mixedSample_re;
        float mixedSample_im;
        float mix_osc;
        float samples_per_cycle;
        float cycles;
        float lookup_step_size;
        int overall_thread_idx;

        samples_per_cycle = (float) (Fs / osc_frequency);
        // printf("Fs is %f and Osc is %f so SPC is %f\n", Fs, osc_frequency, samples_per_cycle);
        cycles = (float) (N / samples_per_cycle);
        // printf("N is %d and SPC is %f so Cycles is %f\n", N, samples_per_cycle ,cycles);
        lookup_step_size = (cycles/ (N/4));
        // printf("Cycles is %f and N is %d so LSS is %f\n", cycles, N, lookup_step_size);
        
        float mixer_angle =  -1 * (index_in*lookup_step_size/2);
        sincospif(mixer_angle, &mixerValue_im, &mixerValue_re);
        mixedSample_re = mixerValue_re * sample_in;
        mixedSample_im = mixerValue_im * sample_in;
        // debug_data_real[index_in + Fir_length] = mixedSample_re;
        // debug_data_imag[index_in + Fir_length] = mixedSample_im;
        debug_data_real[index_in + Fir_length] = mixerValue_re;
        debug_data_imag[index_in + Fir_length] = mixerValue_im;
        

        //1.4 Store in memory. Offset by Fir_length as the first Fir_length samples will be fromthe held back slice from the previous
        mixed_data_re[index_in + Fir_length] = mixedSample_re;
        mixed_data_im[index_in + Fir_length] = mixedSample_im;
        // lookup_state[0] = 1.0; 
    }
    
    // 2. Data has been mixed, now the fir will be applied. Needs to be synced before this happends
    __syncthreads();

    // // 3. Mix Data and store in shared mixed_data array
    float sample_out_re = 0;
    float sample_out_im = 0;

    int base_shared_mixed_sample_index = threadIdx.x * 16 + inOffset; 
    int data_idx = 0;
    for(int i = 0; i < Fir_length; i++){
        data_idx = base_shared_mixed_sample_index - i + (Fir_length-1); // The 255 is added to offset the address so the adressing is flipped (as required for convolution)

        // if (blockIdx.x == 0){
        //     if (threadIdx.x == 1){
        //         printf("%d\n", data_idx);
        //     }
        // }

        float fir_coeff = fir_coeffs_shared[i];
        float mixedSample_re = mixed_data_re[data_idx];
        float mixedSample_im = mixed_data_im[data_idx];

        // if (blockIdx.x == 0){
            // printf("block is %d and i is %d with data_idx is %d and coeff is %f and mix_re is %f\n", blockIdx.x, i,data_idx, fir_coeff, mixedSample_re);
        // }

        sample_out_re = sample_out_re + mixedSample_re * fir_coeff;
        sample_out_im = sample_out_im + mixedSample_im * fir_coeff;

        // if (blockIdx.x ==1){
        //     printf("block is %d and i is %d with data_idx is %d and coeff is %f and sample_out_re is %f\n", blockIdx.x, i,data_idx, fir_coeff, sample_out_re);
        // }
    }

    int index_out = (blockIdx.x*Fir_length + threadIdx.x)*2;

    // debug_data_real[index_out] = sample_out_re;
    // debug_data_imag[index_out] = sample_out_im;

    // printf("index_out is %d and sample_re is %f\n", index_out, sample_out_re);

    data_downsampled_out[index_out] = sample_out_re;
    data_downsampled_out[index_out+1] = sample_out_im;
}