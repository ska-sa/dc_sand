#include <cuComplex.h>

#define SAMPLES_PER_BLOCK 4096 //Limited by the amount of shared memory available per block
#define N 4096
#define Fs 1712e6
#define Fir_length 256

__global__ void kernel_ddc(float *data_in, float *fir_coeffs, float *data_downsampled_out, float osc_frequency, int chunk_number, float *debug_data_real, float *debug_data_imag)
{
    __shared__ float fir_coeffs_shared[Fir_length];
    __shared__ float mixed_data_re[Fir_length + SAMPLES_PER_BLOCK];
    __shared__ float mixed_data_im[Fir_length + SAMPLES_PER_BLOCK];

    //Stage 1: Load Data From memory, mix and store in shared memory
    //1.1 Load FIR coeffs into shared memory
    fir_coeffs_shared[threadIdx.x] = fir_coeffs[threadIdx.x];

    __syncthreads();

    // int inOffset = (blockIdx.x+1)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values
    int inOffset = (blockIdx.x)*(SAMPLES_PER_BLOCK); //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values

    float sample_in = 0;
    int dgb_addr_dst = 0;

    int numLoads = SAMPLES_PER_BLOCK/blockDim.x; // Number of times each thread needs to read input data from global memory. Throws errors whem SAMPLES_PER_BLOCK != 4096 which needs to be investigated
    int numPrevLoads = Fir_length/blockDim.x;//This works out how many samples from the previous block need to be loaded taking into account the FIR size
    if (numPrevLoads == 0)
        numPrevLoads = 1;

    for(int k = -numPrevLoads; k < numLoads; k++){ //The -numPrevLoads accounts for the additional FIR length worth of data that we need to load of past values
        // 1.2 Load Data From Global Memory
        int index_in = inOffset + threadIdx.x + k*Fir_length;
  
        dgb_addr_dst = index_in + 256;

        if (index_in < 0)
        {
            sample_in = 0;
            // printf("block=%d and thread=%d with k=%d and index_in=%d and inoffset=%d with dgb_add= %d\n",blockIdx.x, threadIdx.x, k, index_in, inOffset,dgb_addr_dst);
        }else{
            // printf("block=%d and thread=%d with k=%d and index_in=%d and inoffset=%d with dgb_add= %d\n",blockIdx.x, threadIdx.x, k, index_in, inOffset,dgb_addr_dst);
            sample_in = data_in[index_in];
        }

        
        // The lookup index needs a Fir_length offset as the vector it will be multiplied with is later offset by this amount when appended to the 'mixed_data_re' vector.
        // int lookup_index = index_in + Fir_length;
        int lookup_index = index_in + Fir_length + N*(chunk_number);

        int mix_addr_dst = threadIdx.x + k * Fir_length + 256;

        //1.3 Mix value down
        float mixerValue_re;
        float mixerValue_im;
        float mixedSample_re;
        float mixedSample_im;
        float samples_per_cycle;
        float cycles;
        float lookup_step_size;

        samples_per_cycle = (float) (Fs / osc_frequency);

        lookup_step_size = (float) (1/samples_per_cycle)*2;

        float mixer_angle =  -1 * (lookup_index*lookup_step_size);

        // printf("Lookup\n");
        sincospif(mixer_angle, &mixerValue_im, &mixerValue_re);
        mixedSample_re = mixerValue_re * sample_in;
        mixedSample_im = mixerValue_im * sample_in;

        // debug_data_real[dgb_addr_dst] = sample_in;
        // debug_data_imag[dgb_addr_dst] = sample_in;
        // if (blockIdx.x == 0)
        // {
        //     debug_data_real[mix_addr_dst] = mixedSample_re;
        //     debug_data_imag[mix_addr_dst] = mixedSample_im;
        // }

        // debug_data_real[mix_addr_dst] = mixerValue_re;
        // debug_data_imag[mix_addr_dst] = mixerValue_im;

        //1.4 Store in memory. Offset by Fir_length as the first Fir_length samples will be fromthe held back slice from the previous
        mixed_data_re[mix_addr_dst] = mixedSample_re;
        mixed_data_im[mix_addr_dst] = mixedSample_im;

        // debug_data_real[dgb_addr_dst] = mixedSample_re;
        // debug_data_imag[dgb_addr_dst] = mixedSample_im;

        //# Debug
        // mixed_data_re[mix_addr_dst] = mixerValue_re;
        // mixed_data_im[mix_addr_dst] = mixerValue_im; 
        // mixed_data_re[mix_addr_dst] = sample_in;
        // mixed_data_im[mix_addr_dst] = sample_in;
    }

    // 2. Data has been mixed, now the fir will be applied. Needs to be synced before this happends
    __syncthreads();

    // 3. Mix Data and store in shared mixed_data array
    float sample_out_re = 0;
    float sample_out_im = 0;

    // int base_shared_mixed_sample_index = threadIdx.x * S + inOffset; 
    int base_shared_mixed_sample_index = threadIdx.x * 16; 
    
    int data_idx = 0;
    // 3.1 This for loop is where most of the kernels computation happens. Each iteration of the loop acceses a different location in shared memory. 
    // The kernel is heavily bound by the shared memory bandwidth. One isse is that there are many bank conflicts in the shared memory access in 
    // this loop. Thread 0 acceses element 0. thread 2 acceses element 16, thread 3 acceses element 32, etc. This is almost the worst case shared memory
    // access pattern. By improving this access pattern it may be possible to get a significant improvement in the kernel performance. (I want to say 10x
    // improvement but its difficult to be 100% sure of this.).
   
    for(int i = 0; i < Fir_length; i++){
        data_idx = base_shared_mixed_sample_index - i + (Fir_length-1); // The 255 is added to offset the address so the adressing is flipped (as required for convolution)
        // int dbg_idx = data_idx + (blockIdx.x) * N;

        float fir_coeff = fir_coeffs_shared[i];
        float mixedSample_re = mixed_data_re[data_idx];
        float mixedSample_im = mixed_data_im[data_idx];

        // if (blockIdx.x == 0)
        // {
        //     debug_data_real[data_idx] = mixedSample_re;
        //     debug_data_imag[data_idx] = mixedSample_im;
        // }

        sample_out_re = sample_out_re + mixedSample_re * fir_coeff;
        sample_out_im = sample_out_im + mixedSample_im * fir_coeff;
        
    }

    int index_out = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    int dbg_idx = (blockIdx.x*blockDim.x + threadIdx.x);

    // debug_data_real[dbg_idx] = sample_out_re;
    // debug_data_imag[dbg_idx] = sample_out_im;

    data_downsampled_out[index_out] = sample_out_re;
    data_downsampled_out[index_out+1] = sample_out_im;
    // data_downsampled_out[index_out] = 0;
    // data_downsampled_out[index_out+1] = 0;   
    
}
