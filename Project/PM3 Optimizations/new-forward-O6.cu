//O6 Input channel reduction: tree (3 point)
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define MAX_NUM_THREADS 256
#define MASK_WIDTH 7
#define CHANNEL 4
#define MAP_SIZE 16
#define SM_IN (TILE_WIDTH + MASK_WIDTH - 1)
__constant__ float Const_Mask [MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_WIDTH];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    __shared__ float SM_Input [CHANNEL*SM_IN*SM_IN];
    __shared__ float SM_Output [CHANNEL][TILE_WIDTH][TILE_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int m = blockIdx.x;
    int h_offset = (blockIdx.y / W_size) * TILE_WIDTH;
    int w_offset = (blockIdx.y % W_size) * TILE_WIDTH;
    int b = blockIdx.z;
    //100, 4, 1, 86, 86, 7
    //100, 16, 4, 40, 40, 7

    const int BlockSize = TILE_WIDTH * TILE_WIDTH;
    const int InputSize = Batch * Channel * Height * Width;
    const int msize = Channel * K * K;
    const int SM_Width = TILE_WIDTH + K - 1;
    const int SM_ChSize = SM_Width * SM_Width;
    const int SM_InputSize = Channel * SM_ChSize;

    #define mask_4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (K * K) + (h) * (K) + w]
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    #define input_idx(b, c, h, w) ((b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w)
    #define sm_in_idx(c, h, w) ((c)*(SM_ChSize) + (h)*(SM_Width) + w)

    SM_Output[tz][ty][tx] = 0;
    __syncthreads();

    // shared memory subinput
    for(int newIdx = ty*TILE_WIDTH + tx; newIdx < SM_ChSize; newIdx += BlockSize){
        int newY = newIdx / SM_Width;
        int newX = newIdx % SM_Width;
        int index = input_idx(b,tz,newY+h_offset,newX+w_offset);
        if(index < InputSize)
            SM_Input[sm_in_idx(tz,newY,newX)] = input[index];
    }
    __syncthreads();

    // if(m == 0 && b == 0 && h_offset == 0 && w_offset == 0){
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
    int h = h_offset + ty, w = w_offset + tx;
    if(h < Height_out && w < Width_out){
        float acc = 0.0f;
        for (int p = 0; p < K; p++){ // loop over KxK filter
            for (int q = 0; q < K; q++){
                acc += SM_Input[sm_in_idx(tz, ty+p, tx+q)] * mask_4d(m, tz, p, q);
            }
        }
        // int offset = tz*TILE_WIDTH*TILE_WIDTH + ty*TILE_WIDTH + tx;
        // SM_Output[offset] = acc;
        SM_Output[tz][ty][tx] = acc;
        __syncthreads();

        // if(tz == 0){
        //     for(int i = 1; i < Channel; i++){
        //         SM_Output[0][ty][tx] += SM_Output[i][ty][tx];
        //     }
        // }

        //Restriction: Channel is power of 2
        for (unsigned int stride = Channel / 2; stride >= 1; stride /= 2) {
            __syncthreads();
            if (tz < stride && tz + stride < Channel){
                SM_Output[tz][ty][tx] += SM_Output[tz+stride][ty][tx];
                // SM_Output[offset] += SM_Output[offset + stride * TILE_WIDTH * TILE_WIDTH];
            }
        }
        __syncthreads();
        if(tz == 0){
            out_4d(b, m, h, w) = SM_Output[0][ty][tx];
        }
    }

    #undef out_4d
    #undef mask_4d
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

    cudaMemcpyToSymbol(Const_Mask, host_mask, Map_out * Channel * K * K * sizeof(float));

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
    int W_size = ceil(1.0 * W_out / TILE_WIDTH);
    int H_size = ceil(1.0 * H_out / TILE_WIDTH);

    dim3 DimGrid(Map_out,W_size*H_size,Batch);
    // dim3 DimGrid(1,W_size*H_size,Batch);
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,Channel);
    // Set the kernel dimensions and call the kernel
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
