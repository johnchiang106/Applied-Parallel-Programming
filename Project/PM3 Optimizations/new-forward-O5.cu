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

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // __shared__ float SM_Input [CHANNEL*SM_IN*SM_IN];
    __shared__ __half SM_Input [CHANNEL*SM_IN*SM_IN];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);
    //100, 4, 1, 86, 86, 7
    //100, 16, 4, 40, 40, 7

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.x;
    int h_offset = (blockIdx.y / W_size) * TILE_WIDTH;
    int w_offset = (blockIdx.y % W_size) * TILE_WIDTH;
    int b = blockIdx.z;

    const int InputSize = Batch * Channel * Height * Width;
    const int SM_Width = TILE_WIDTH + K - 1;
    const int msize = Channel * K * K;

    #define mask_4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (K * K) + (h) * (K) + w]
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    #define input_idx(b, c, h, w) ((b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w)
    #define sm_in_idx(c, h, w) ((c)*(SM_Width*SM_Width) + (h)*(SM_Width) + w)

    // shared memory subinput
    // SM_Width = TILE_WIDTH + K - 1;
    for (int c = 0; c < Channel; c++) {
        for (int j = ty; j < SM_Width; j += TILE_WIDTH){
            for (int k = tx; k < SM_Width; k += TILE_WIDTH){
                int sm_index = sm_in_idx(c,j,k);
                int index = input_idx(b,c,j+h_offset,k+w_offset);
                if(index < InputSize)
                    SM_Input[sm_index] = __float2half(input[index]);
            }
        }
    }
    __syncthreads();
    
    if(h_offset + ty < Height_out && w_offset + tx < Width_out){
        // float acc = 0.0f;
        __half acc = __float2half(0.0f);
        for (int c = 0; c < Channel; c++) { // sum over all input channels
            for (int p = 0; p < K; p++){ // loop over KxK filter
                for (int q = 0; q < K; q++){
                    __half halfMask = __float2half(mask_4d(m, c, p, q));
                    acc = __hadd(acc, __hmul(SM_Input[sm_in_idx(c, ty+p, tx+q)], halfMask));
                    // acc += SM_Input[sm_in_idx(c, ty+p, tx+q)] * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h_offset + ty, w_offset + tx) = acc;
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
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
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
