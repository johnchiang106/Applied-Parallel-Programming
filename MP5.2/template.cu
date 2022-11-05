// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
__global__ void add(float *input, float *output, int len) {
  __shared__ float sharedMemory[BLOCK_SIZE * 2];
  int bx = blockIdx.x, tx = threadIdx.x, index = bx * BLOCK_SIZE * 2 + tx;

  if (index < len)  sharedMemory[tx] = input[index];

  if (bx > 0){
    float prevSum = 0.0;
    for (int i = 1; i <= bx; i++)
      prevSum += input[i * 2 * BLOCK_SIZE - 1];

    sharedMemory[tx] += prevSum;
  }
  if (index < len)  output[index] = sharedMemory[tx];
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float sharedMemory[BLOCK_SIZE * 2];
  int bx = blockIdx.x, tx = threadIdx.x,
   x1 = bx * BLOCK_SIZE * 2 + tx, x2 = x1 + BLOCK_SIZE;

  if (x1 < len) sharedMemory[tx] = input[x1];
  else sharedMemory[tx] = 0;

  if (x2 < len)
    sharedMemory[tx + BLOCK_SIZE] = input[x2];
  else sharedMemory[tx + BLOCK_SIZE] = 0;

  __syncthreads();

  for(int stride = 1; stride < 2 * BLOCK_SIZE; stride <<= 1) {
    int index = (tx + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
      sharedMemory[index] += sharedMemory[index-stride];
    __syncthreads();
  }

  for(int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) 
      sharedMemory[index + stride] += sharedMemory[index];
    __syncthreads();
  }

  if (x1 < len) output[x1] = sharedMemory[tx];
  if (x2 < len) output[x2] = sharedMemory[tx + BLOCK_SIZE];

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numElements/BLOCK_SIZE/2,1,1);
  if (numElements%(BLOCK_SIZE*2)) DimGrid.x++;
  dim3 DimBlock(BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan on the deivce
  scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, numElements);
  DimBlock.x <<= 1;
  add<<<DimGrid,DimBlock>>>(deviceOutput, deviceInput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
