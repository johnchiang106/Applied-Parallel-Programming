#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 4
//@@ Define constant memory for device kernel here
// __constant__ float MASK[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
__constant__ float MASK[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float sharedMemoryTile [TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
 
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  
  // Identify the address of the input tile element in shared memory
  int z_in = bz * TILE_WIDTH + tz;
  int y_in = by * TILE_WIDTH + ty;
  int x_in = bx * TILE_WIDTH + tx;
  
  // copy input to shared memory
  if (z_in >= 0 && z_in < z_size && 
      y_in >= 0 && y_in < y_size && 
      x_in >= 0 && x_in < x_size)
    sharedMemoryTile[tz][ty][tx] = 
    input[y_size*x_size*z_in + x_size*y_in + x_in];
  else
    sharedMemoryTile[tz][ty][tx] = 0;
  
  __syncthreads();
  
  // do convolution for every element of the output tile
  float res = 0.0f;
  
  for (int x = 0; x < MASK_WIDTH; x++)
  for (int y = 0; y < MASK_WIDTH; y++)
  for (int z = 0; z < MASK_WIDTH; z++)
  {
    float value = 0;
    
    // find corresponding index in shared memory tile
    int z_tile = tz + z - MASK_RADIUS;
    int y_tile = ty + y - MASK_RADIUS;
    int x_tile = tx + x - MASK_RADIUS;
    
    // if index is inside the range of the tile - use shared memory
    if (z_tile >= 0 && z_tile < TILE_WIDTH &&
        y_tile >= 0 && y_tile < TILE_WIDTH &&
        x_tile >= 0 && x_tile < TILE_WIDTH)
      value = sharedMemoryTile[z_tile][y_tile][x_tile];
    // if index is outside - take value from global memory (except when     outside the range)
    else{
      // find corresponding index in global memory
      int z_global = z_in + z - MASK_RADIUS;
      int y_global = y_in + y - MASK_RADIUS;
      int x_global = x_in + x - MASK_RADIUS;
      
      if (z_global >= 0 && z_global < z_size &&
          y_global >= 0 && y_global < y_size &&
          x_global >= 0 && x_global < x_size)
        value = input[y_size*x_size*z_global + x_size*y_global + x_global];
    }
  
  //when value is known - add it the corresponding output value
    res += value * MASK[z*9 + y*3 + x];
  }
  if (z_in < z_size && y_in < y_size && x_in < x_size)
    output[x_size * y_size * z_in + x_size * y_in + x_in] = res;
  
  __syncthreads();

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  cudaMalloc((void **) &deviceInput, z_size * y_size * x_size * sizeof(float));
  cudaMalloc((void **) &deviceOutput, z_size * y_size * x_size * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(MASK, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(x_size/TILE_WIDTH,y_size/TILE_WIDTH,z_size/TILE_WIDTH);
  if (x_size%TILE_WIDTH)  DimGrid.x++;
  if (y_size%TILE_WIDTH)  DimGrid.y++;
  if (z_size%TILE_WIDTH)  DimGrid.z++;
  dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, z_size * y_size * x_size * 
sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
