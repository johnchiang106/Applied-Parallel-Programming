//O4 Shared memory matrix multiplication and input matrix unrolling (3 points)
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

__global__ void unroll_Kernel(int B, int C, int H, int W, int K, float* X, float* X_unroll){
    //Batch * Channel * H_size * W_size * K * K
    #define X(b, c, h, w) X[(b)*(C*H*W) + (c)*(H*W) + (h)*(W) + w]
    #define X_unroll(base, pos, s) X_unroll[base + (pos)*(unroll_size) + s]
    int b, c, s, h, w, w_base, p, q;
    int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
    int H_out = H - K + 1; //xx
    int W_out = W - K + 1;
    int unroll_size = H_out * W_out;
    if (t < B * C * unroll_size) {
        b = t / (C * unroll_size);
        c = t / unroll_size;
        s = t % unroll_size;
        h = s / W_out;
        w = s % W_out;
        w_base = (b * C + c) * K * K * unroll_size;
        for(p = 0; p < K; p++)
            for(q = 0; q < K; q++) {
                // int pos = w_base + p * K + q;
                int pos = p * K + q;
                //Batch * Channel * H_size * W_size
                X_unroll(w_base, pos, s) = X(b, c, h + p, w + q);
            }
    }

    // int offset = ty*TILE_WIDTH + tx;
    // if(h < Height_out && w < Width_out){
    //     for (int c = 0; c < Channel; c++){ // sum over all input channels
    //         for (int p = 0; p < K; p++){ // loop over KxK filter
    //             for (int q = 0; q < K; q++){
    //                 SM_Unroll[offset] = SM_Input[sm_in_idx(c, ty+p, tx+q)];
    //                 offset += BlockSize;
    //             }
    //         }
    //     }
    // }
}

__global__ void conv_forward_kernel(float *output, float *unroll, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    // __shared__ float SM_Input [CHANNEL*SM_IN*SM_IN];
    // __shared__ float SM_Unroll [TILE_WIDTH*TILE_WIDTH*MASK_WIDTH*MASK_WIDTH*CHANNEL];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_size = Width_out/TILE_WIDTH;
    // int H_size = Height_out/TILE_WIDTH;
    if (Width_out%TILE_WIDTH)  W_size++;
    // if (Height_out%TILE_WIDTH)  H_size++;
    // int W_size = ceil((Width_out * 1.0) / TILE_WIDTH);
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.x;
    int b = blockIdx.z;

    const int unroll_size = Height_out * Width_out;
    const int msize = Channel * K * K;
    const int BlockSize = TILE_WIDTH * TILE_WIDTH;

    #define in_4d(b, c, h, w) input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w]
    #define mask_4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (K * K) + (h) * (K) + w]
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    
    // if(m == 0 && b == 0 && h == 0 && w == 0){
    //     printf("%i, %i, %i, %i, %i, %i\n", Batch, Map_out, Channel, Height, Width, K);
    // }
    // __syncthreads();
    //100, 4, 1, 86, 86, 7
    //100, 16, 4, 40, 40, 7

    // unroll
    // TILE_WIDTH*TILE_WIDTH*K*K*Channel
    // int offset = ty*TILE_WIDTH + tx;
    // if(h < Height_out && w < Width_out){
    //     for (int c = 0; c < Channel; c++){ // sum over all input channels
    //         for (int p = 0; p < K; p++){ // loop over KxK filter
    //             for (int q = 0; q < K; q++){
    //                 SM_Unroll[offset] = SM_Input[sm_in_idx(c, ty+p, tx+q)];
    //                 offset += BlockSize;
    //             }
    //         }
    //     }
    // }
    // __syncthreads();

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
    
    // if(h < Height_out && w < Width_out){
        float acc = 0.0f;
        int base = Channel * K * K * unroll_size;
        for(int k = ty*TILE_WIDTH + tx; k < unroll_size; k += BlockSize){
            // int i = 0;
            // int index = b * base + k;
            for(int i = 0, index = b * base + k; i < msize; i++, index += unroll_size){
                acc += unroll[index] * mask_4d(m, 0, 0, i);
            }
            out_4d(b, m, k/Width_out, k%Width_out) = acc;
        }
    // }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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

    // Unroll Input
    float *device_unroll;
    cudaMalloc((void **) &device_unroll, Batch * Channel * H_size * W_size * K * K * sizeof(float));
    
    // int num_threads = Batch * Channel * H_out * W_out;
    int num_blocks = ceil((Batch * Channel * H_out * W_out * K * K) / MAX_NUM_THREADS);
    dim3 grid_unroll(num_blocks,1,1);
    unroll_Kernel<<<num_blocks, MAX_NUM_THREADS>>>(Batch, Channel, Height, Width, K, device_input, device_unroll);

    // printf("unroll finished!\n");

    dim3 DimGrid(Map_out,1,Batch);
    // dim3 DimGrid(1,W_size*H_size,Batch);
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
    // Set the kernel dimensions and call the kernel
    conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_unroll, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
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
