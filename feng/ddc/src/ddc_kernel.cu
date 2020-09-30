#include <cuComplex.h>

#define N 8192*4
#define Fs 1712e6
#define Fir_length 256

__device__ float mixed_data_re[Fir_length + N];
__device__ float mixed_data_im[Fir_length + N];
// __device__ bool run_once = true;  

__global__ void kernel_ddc(float *data_in, float *fir_coeffs, float *data_downsampled_out, float osc_frequency, int chunk_number, float *debug_data_real, float *debug_data_imag)
{
    __shared__ float fir_coeffs_shared[Fir_length];
    // __shared__ float prev_data_slice_re[Fir_length];
    // __shared__ float prev_data_slice_im[Fir_length];
    // __shared__ float mixed_data_re[256+256];
    // __shared__ float mixed_data_im[256+256];
    __syncthreads();

    //Stage 1: Load Data From memory, mix and store in shared memory
    //1.1 Load FIR coeffs into shared memory
    fir_coeffs_shared[threadIdx.x] = fir_coeffs[threadIdx.x];

    // printf("Chunk Number is %d\n", chunk_number);

    // if ((blockIdx.x == 0)&(run_once == true)){
    //     for(int i = 0; i <= (Fir_length + N); i++){
    //         debug_data_real[i] = 0;
    //         debug_data_imag[i] = 0;
    //     }
    //     run_once = false;
    // }

    __syncthreads();

    if (blockIdx.x == 0){
        // count = count + 1;
        // if (chunk_number == 63)
        // {

        //     printf("Count is %d chunk is %d\n", count, chunk_number);
        // }

        // 1.2 Copy the trailing data (of Fir_length) to the beginning of the new data word that has been received.
        int dest_addr = Fir_length - threadIdx.x - 1;
        int src_addr = Fir_length + N - threadIdx.x - 1;

        mixed_data_re[dest_addr] = mixed_data_re[src_addr];
        mixed_data_im[dest_addr] = mixed_data_im[src_addr];
        // if (src_addr > 32000){
        // printf("Blockx.x is %d and Threadx.x is %d so dest %d from %d\n", blockIdx.x, threadIdx.x, dest_addr, src_addr);
        // }
        debug_data_real[dest_addr] = mixed_data_re[src_addr];
        debug_data_imag[dest_addr] = mixed_data_im[src_addr];
    }
    __syncthreads();

    // int inOffset = (blockIdx.x+1)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values
    int inOffset = (blockIdx.x)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values

    for(int i = 0; i <= 15; i++){ //The minus 1 accounts for the additional FIR length worth of data that we need to load of past values
    // for(int i = -1; i < 16; i++){ //The minus 1 accounts for the additional FIR length worth of data that we need to load of past values
        // 1.2 Load Data From Global Memory

        int index_in = inOffset + threadIdx.x + i*Fir_length;
        int lookup_index = index_in + chunk_number*N;
        // int lookup_index = index_in;
        float sample_in = data_in[index_in];

        // printf("Test\n");

        // if (index_in > 16380 & index_in < 16400){
        //     printf("Index_in is %d\n", index_in);
        // }

        //1.3 Mix value down
        float mixerValue_re;
        float mixerValue_im;
        float mixedSample_re;
        float mixedSample_im;
        // float mix_osc;
        float samples_per_cycle;
        float cycles;
        float lookup_step_size;
        // int overall_thread_idx;

        samples_per_cycle = (float) (Fs / osc_frequency);
        // printf("Fs is %f and Osc is %f so SPC is %f\n", Fs, osc_frequency, samples_per_cycle);
        cycles = (float) (N / samples_per_cycle);
        // printf("N is %d and SPC is %f so Cycles is %f\n", N, samples_per_cycle ,cycles);
        lookup_step_size = (cycles/ (N/4));
        // printf("Cycles is %f and N is %d so LSS is %f\n", cycles, N, lookup_step_size);
        
        float mixer_angle =  -1 * (lookup_index*lookup_step_size/2);
        if (lookup_index == 16128)
        {
            printf("lookup_index is %d with lookup_step_size of %f and Mixer Angle is %f\n", lookup_index, lookup_step_size, mixer_angle);
        }

        sincospif(mixer_angle, &mixerValue_im, &mixerValue_re);
        mixedSample_re = mixerValue_re * sample_in;
        mixedSample_im = mixerValue_im * sample_in;

        int mix_addr_dst = index_in + Fir_length;

        // if (mix_addr_dst > 30000){
        //     printf("mix_addr_dst is %d and mixedSample_re if %f\n", mix_addr_dst, mixedSample_re);
        // }

        debug_data_real[mix_addr_dst] = mixedSample_re;
        debug_data_imag[mix_addr_dst] = mixedSample_im;
        // debug_data_real[index_in + Fir_length] = mixerValue_re;
        // debug_data_imag[index_in + Fir_length] = mixerValue_im;

        //1.4 Store in memory. Offset by Fir_length as the first Fir_length samples will be fromthe held back slice from the previous
        mixed_data_re[mix_addr_dst] = mixedSample_re;
        mixed_data_im[mix_addr_dst] = mixedSample_im;
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
        data_idx = base_shared_mixed_sample_index - i + (Fir_length); // The 255 is added to offset the address so the adressing is flipped (as required for convolution)

        // if (blockIdx.x == 0){
        //     if (threadIdx.x == 1){
        //         printf("%d\n", data_idx);
        //     }
        // }

        // if (data_idx > 33000)
        // {
        //     printf("data_idx is %d\n", data_idx);
        // }

        float fir_coeff = fir_coeffs_shared[i];
        float mixedSample_re = mixed_data_re[data_idx];
        float mixedSample_im = mixed_data_im[data_idx];

        // debug_data_real[data_idx] = data_idx;
        // debug_data_imag[i] = i;

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

    // int debug_index_out = (blockIdx.x*Fir_length + threadIdx.x);

    // debug_data_real[debug_index_out] = sample_out_re;
    // debug_data_imag[debug_index_out] = sample_out_im;

    // printf("index_out is %d and sample_re is %f\n", index_out, sample_out_re);

    data_downsampled_out[index_out] = sample_out_re;
    data_downsampled_out[index_out+1] = sample_out_im;
}