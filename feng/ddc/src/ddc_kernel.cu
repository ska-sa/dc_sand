#include <cuComplex.h>

// #define N 1048576
#define N 8192*2
#define Fs 1712e6
#define D 16

__device__ float mixed_data_re[N];
__device__ float mixed_data_im[N];  

__global__ void kernel_ddc(float *data_in, float *fir_coeffs, float *data_downsampled_out, float osc_frequency, float *debug_data_real, float *debug_data_imag)
{
    __shared__ float fir_coeffs_shared[256+256];
    // __shared__ float mixed_data_re[256+256];
    // __shared__ float mixed_data_im[256+256];
 

    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    // float N = 8192.0;

    // printf("blockDim.x is %d\n", blockDim.x);
    // printf("blockIdx.x is %d\n", blockIdx.x);
    // printf("threadIdx.x is %d\n", threadIdx.x);
    // printf("threadId is %d\n", threadId);

    //Stage 1: Load Data From memory, mix and store in shared memory

    //1.1 Load FIR coeffs into shared memory
    fir_coeffs_shared[threadIdx.x] = fir_coeffs[threadIdx.x];

    // for(int i = 0; i < 17; i++){ 
    
    int inOffset = (blockIdx.x+1)*4096; //The plus 1 is because we skip the first block of data: assuming that it is wasted as we cannot access past values
    for(int i = -1; i < 16; i++){ //The minus 1 accounts for the additional FIR length worth of data that we need to load of past values
        // 1.2 Load Data From GLobal Memory
        int index_in = inOffset + threadIdx.x + i*256;
        float sample_in = data_in[index_in];
        // printf("%d %d %d %d %d\n",threadId,blockIdx.x,threadIdx.x,i,index_in);

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

        // overall_thread_idx = threadIdx.x + (blockIdx.x * 256) + (i * 256);

        // int n = overall_thread_idx;
        int n = index_in;
        // if (n < 4096){
        //     printf("%d\n",n);
        // }

        samples_per_cycle = (float) (Fs / osc_frequency);
        // printf("samples_per_cycle %f\n",samples_per_cycle);
        cycles = (float) (N / samples_per_cycle);
        // printf("cycles %f\n",cycles);
        lookup_step_size = cycles/N;
        // printf("N is %d\n",N);
        // printf("Lookup Step Size: %f\n", lookup_step_size);

        // float mixer_angle =  -2.0 * osc_frequency/((float) Fs) * n/((float) N) ; //~45 degrees, hardcoded for now, need to calculate it
        float mixer_angle =  -1 * (n*lookup_step_size/2);
        // printf("Mixer Angle %f\n", mixer_angle);

        sincospif(mixer_angle, &mixerValue_im, &mixerValue_re);
        // mix_osc = exp(mixer_angle);
        // printf("Block: %d with Thread %d has Mix Osc %f\n:", blockIdx.x, overall_thread_idx, mix_osc);
        // debug_data[threadIdx.x] = mix_osc;
        // debug_data_real[n] = mixerValue_re;
        // debug_data_imag[n] = mixerValue_im;
        mixedSample_re = mixerValue_re * sample_in;
        mixedSample_im = mixerValue_im * sample_in;
        debug_data_real[n] = mixedSample_re;
        debug_data_imag[n] = mixedSample_im;

        // printf("mixerValue_re is %f\n", mixerValue_re);
        // printf("mix_osc is %f\n", mix_osc);

        //1.4 Store in shared memoy
        // mixed_data_re[threadIdx.x + i*256] = mixedSample_re;
        // mixed_data_im[threadIdx.x + i*256] = mixedSample_im;
        mixed_data_re[n] = mixedSample_re;
        mixed_data_im[n] = mixedSample_im;


        // if (blockIdx.x == 0){
        //     if (threadIdx.x == 1){
        //         printf("n is %d and mixed_re is %f\n", n, mixed_data_re[n]);
        //     }
        // }



        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("f %f n %d N %d\n",osc_frequency, n, N);
        //     printf("i %d,theta = %.12f cos(theta) = %.12f sin(theta) = %0.12f\n",index_in,mixer_angle,mixerValue_re, mixerValue_im);
        //     printf("Mixed Sample Re %f Mixed Sample Im %f\n",mixedSample_re,mixedSample_im);
        // }
    }
    
    // 2. Data has been mixed, now the fir will be applied. Needs to be synced before this happends
    __syncthreads();

    // 3. Mix Data and store in shared mixed_data array
    float sample_out_re = 0;
    float sample_out_im = 0;

    // printf("Decimated length is %d\n", N/D);
    // int base_shared_mixed_sample_index = 256 + threadIdx.x * 16; 
    
    // for(int i = 0; i < 256; i++){
    //     sample_out_re = sample_out_re + mixed_data_re[n] * fir_coeffs_shared[i];
    //     sample_out_im = sample_out_im + mixed_data_im[n] * fir_coeffs_shared[i];
    // }

    // sample_out_re = sample_out_re + mixedSample_re * fir_coeff;
    // sample_out_im = sample_out_im + mixedSample_im * fir_coeff;

    int base_shared_mixed_sample_index = threadIdx.x * 16 + inOffset; 
    int data_idx = 0;
    for(int i = 0; i < 256; i++){
        data_idx = base_shared_mixed_sample_index - i;

        // if (data_idx >= 0){
            // if (blockIdx.x == 0){
            //     if (threadIdx.x == 3){
            //         printf("data_idx is %d\n", data_idx);
            //     }
            // }

            float fir_coeff = fir_coeffs_shared[i];
            float mixedSample_re = mixed_data_re[data_idx];
            float mixedSample_im = mixed_data_im[data_idx];

            // if (blockIdx.x == 0){
            //     if (threadIdx.x == 1){
            //         printf("i is %d and data_idx is %d so coeff is %f and data is %f\n", i, data_idx, fir_coeff, mixedSample_re);
            //     }
            // }
    
            sample_out_re = sample_out_re + mixedSample_re * fir_coeff;
            sample_out_im = sample_out_im + mixedSample_im * fir_coeff;
        // }
    }

    // int index_out = ((blockIdx.x+1) * 256 + threadIdx.x) * 2;
    int index_out = (blockIdx.x*256 + threadIdx.x)*2;


    // printf("index_out is %d\n", index_out);

    data_downsampled_out[index_out] = sample_out_re;
    data_downsampled_out[index_out+1] = sample_out_im;
}