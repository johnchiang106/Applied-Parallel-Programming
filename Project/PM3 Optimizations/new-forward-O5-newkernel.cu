//O5 Fixed point (FP16) arithmetic. (note this can modify model accuracy slightly) (4 points)
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16
#define MAX_NUM_THREADS 256
#define MASK_WIDTH 7
#define CHANNEL 4
#define MAP_SIZE 16
#define SM_IN (TILE_WIDTH + MASK_WIDTH - 1)
__constant__ float Const_Mask [MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_WIDTH];
// __constant__ __half Half_Mask [MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_WIDTH];

// __global__ void 2half(){
//     int index = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
//     Half_Mask[index] = __float2half(Const_Mask[index]);
// }

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    // __shared__ float SM_Mask [CHANNEL*MASK_WIDTH*MASK_WIDTH];
    __shared__ __half SM_Input [CHANNEL*SM_IN*SM_IN];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_size = Width_out/TILE_WIDTH;
    if (Width_out%TILE_WIDTH)  W_size++;

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + ty; //xx
    int w = (blockIdx.y % W_size) * TILE_WIDTH + tx;
    int b = blockIdx.z;

    const int InputSize = Batch * Channel * Height * Width;
    const int SM_Width = TILE_WIDTH + K - 1;
    // const int BlockSize = TILE_WIDTH * TILE_WIDTH;
    const int msize = Channel * K * K;

    #define in_4d(b, c, h, w) input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w]
    #define mask_4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (K * K) + (h) * (K) + w]
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    // #define sm_mask_3d(c, h, w) SM_Mask[(c)*(K*K) + (h)*(K) + w]
    #define input_idx(b, c, h, w) ((b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w)
    #define sm_in_idx(c, h, w) ((c)*(SM_Width*SM_Width) + (h)*(SM_Width) + w)
    //100, 4, 1, 86, 86, 7
    //100, 16, 4, 40, 40, 7

    // shared memory subinput
    // SM_Width = TILE_WIDTH + K - 1;
    for (int c = 0; c < Channel; c++) {
        for (int j = 0; j < SM_Width; j += TILE_WIDTH){
            for (int k = 0; k < SM_Width; k += TILE_WIDTH){
                int x_in = k + tx, y_in = j + ty;   //xx
                if(x_in < SM_Width && y_in < SM_Width){
                    int sm_index = sm_in_idx(c,y_in,x_in);
                    y_in += (blockIdx.y / W_size) * TILE_WIDTH;
                    x_in += (blockIdx.y % W_size) * TILE_WIDTH;
                    int index = input_idx(b,c,y_in,x_in);
                    if(sm_index < (Channel*SM_Width*SM_Width) && index < InputSize) //xx
                        SM_Input[sm_index] = __float2half(input[index]);
                    // else
                    //     SM_Input[sm_index] = 0;
                }
            }
        }
    }
    __syncthreads();

    // if(m == 0 && b == 0 && h == 0 && w == 0){
    //     int s = Map_out * Channel * K * K;
        // for(int i = 0; i < s; ++i){
        //     printf("%f, ", mask[i]);
        // }
        // printf("\n");
    //     s = MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_WIDTH;
    //     for(int i = 0; i < s; ++i){
    //         printf("%f, ", Const_Mask[i]);
    //     }
    //     printf("\n");
    // }
    // __syncthreads();
    
    if(h < Height_out && w < Width_out){
        // for (int m = 0; m < Map_out; m++) {
            // float acc = 0.0f;
            __half acc = __float2half(0.0f);
            for (int c = 0; c < Channel; c++) { // sum over all input channels
                for (int p = 0; p < K; p++){ // loop over KxK filter
                    for (int q = 0; q < K; q++){
                        // __half halfIn = __float2half(SM_Input[sm_in_idx(c, ty+p, tx+q)]);
                        __half halfMask = __float2half(mask_4d(m, c, p, q));
                        acc = __hadd(acc, __hmul(SM_Input[sm_in_idx(c, ty+p, tx+q)], halfMask));
                    }
                }
            }
            out_4d(b, m, h, w) = __half2float(acc);
        // }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    // #undef sm_mask_3d
    #undef input_idx
    #undef sm_in_idx
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W_out = Width - K + 1, H_out = Height - K + 1;
    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * W_out * H_out * sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers, which are passed to the other two functions.
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(Const_Mask, host_mask, Map_out * Channel * K * K * sizeof(float));
    // for(int i = 0; i < s; ++i){
    //     printf("%f, ", Const_Mask[i]);
    // }
    
    // int num_threads = Batch * Channel * H_out * W_out;
    // int num_blocks = ceil(1.0 * (Map_out * Channel * K * K) / MAX_NUM_THREADS);
    // dim3 grid_unroll(num_blocks,1,1);
    // 2half<<<num_blocks, MAX_NUM_THREADS>>>();

    // }
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W_out = Width - K + 1, H_out = Height - K + 1;
    int W_size = W_out/TILE_WIDTH, H_size = H_out/TILE_WIDTH;
    if (W_out%TILE_WIDTH)  W_size++;
    if (H_out%TILE_WIDTH)  H_size++;


    dim3 DimGrid(Map_out,W_size*H_size,Batch);
    // dim3 DimGrid(1,W_size*H_size,Batch);
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
    // Set the kernel dimensions and call the kernel
    // conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_unroll, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int W_out = Width - K + 1, H_out = Height - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * W_out * H_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
