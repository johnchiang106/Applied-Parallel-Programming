// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here
__global__ void rgb2grey(float *input, unsigned char *outputRGB, unsigned char *outputGrey, int width, int height, int channels){
  int x_in = blockIdx.x * blockDim.x + threadIdx.x;
  int y_in = blockIdx.y * blockDim.y + threadIdx.y;
  if(x_in < width && y_in < height){
    int index = (width * y_in + x_in) * channels;
    unsigned char r = (unsigned char) (255 * input[index]);
    unsigned char g = (unsigned char) (255 * input[index + 1]);
    unsigned char b = (unsigned char) (255 * input[index + 2]);
    outputRGB[index] = r;
    outputRGB[index + 1] = g;
    outputRGB[index + 2] = b;
    outputGrey[width * y_in + x_in] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histogram(unsigned char *input, float *output, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_s[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < len) {
    int pos = input[idx];
    atomicAdd(&(histo_s[pos]), 1);
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histo_s[threadIdx.x]);
  }
  __syncthreads();
  int count = 0;
  if(idx == 0){
    for(int i = 0; i < HISTOGRAM_LENGTH; ++i){
      count += output[i];
    }
    printf("%i, %i \n", count, len);
  }
  __syncthreads();

  if(idx < 256) {
    // printf("%f, ", histPtr[indX]);
    output[idx] /= (float)(len);
  }
  // __syncthreads();
  // if(idx == 0){
  //   for(int i = 0; i < HISTOGRAM_LENGTH; ++i){
  //     printf("%i, %f \n", i, output[i]);
  //   }
  // }
}

__global__ void grey2histogram(unsigned char *input, float *histPtr, int width, int height){
  // __shared__ float sharedMemory[HISTOGRAM_LENGTH];
  int bx = blockIdx.x, dx = blockDim.x, tx = threadIdx.x;
  int indX = bx * dx + tx;
  // int x_in = bx * dx * 2 + tx;
  
  //if (indX==0) printf("val: %d, hist: %f \n",input[indX], histPtr[input[indX]]);
  // atomicAdd( &(histPtr[input[indX]]), 1);
  if (indX < HISTOGRAM_LENGTH) {
    histPtr[indX] = 0;
  }
  __syncthreads();
  if (indX < width * height) {
    atomicAdd( &(histPtr[input[indX]]), 1);
  }
  
  //if (indX==0) printf("val: %d, hist: %f \n",input[indX], histPtr[input[indX]]);
  __syncthreads();
  int count = 0;
  if(indX == 0){
    for(int i = 0; i < HISTOGRAM_LENGTH; ++i){
      count += histPtr[i];
    }
    printf("%i, %i \n", count, width*height);
  }
  __syncthreads();
  if(indX < 256) {
    // printf("%f, ", histPtr[indX]);
    histPtr[indX] /= (float)(width*height);
  }

  __syncthreads();
  
  // if(indX == 0){
  //   for(int i = 0; i < HISTOGRAM_LENGTH; ++i){
  //     printf("%i, %f \n", i, histPtr[i]);
  //   }
  // }
}

__global__ void scan(float *input, float *output) {
  
  __shared__ float sharedMemory[HISTOGRAM_LENGTH];
  int bx = blockIdx.x; int tx = threadIdx.x;
  int x_in = bx * blockDim.x * 2 + tx;
  
  if (x_in < HISTOGRAM_LENGTH) sharedMemory[tx] = input[x_in];
  else sharedMemory[tx] = 0;
  
  if (x_in + blockDim.x < HISTOGRAM_LENGTH) sharedMemory[tx + blockDim.x] = input[x_in + blockDim.x];
  else sharedMemory[tx + blockDim.x] = 0;
  
  __syncthreads();
  
  for (int stride = 1; stride <= blockDim.x; stride *= 2){
    int index = (tx+1) * stride * 2 - 1; 
    if(index < 2 * blockDim.x) sharedMemory[index] += sharedMemory[index-stride];
    __syncthreads();
  }
  
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
    int index = (tx+1) * stride * 2 - 1;
    if(index + stride < 2 * blockDim.x) sharedMemory[index + stride] += sharedMemory[index];
    __syncthreads();
  }
  
  if (x_in < HISTOGRAM_LENGTH) output[x_in] = sharedMemory[tx];
  if (x_in + blockDim.x < HISTOGRAM_LENGTH) output[x_in+blockDim.x] = sharedMemory[tx+blockDim.x];
  __syncthreads();

  // if(x_in == 0){
  //   for(int i = 0; i < HISTOGRAM_LENGTH; ++i)
  //     printf("%i, %f \n", i, output[i]);
  // }
}

__global__ void equalize(float *cdf, unsigned char *inout, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    float equalized = 255.0*(cdf[inout[id]]-cdf[0])/(1.0-cdf[0]);
    inout[id] = (unsigned char) (min(max(equalized, 0.0), 255.0));
  }
}

__global__ void equalization(float *cdf, unsigned char *image, int width, int height, int channels){
  int indX = blockIdx.x * blockDim.x + threadIdx.x;
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  // if(indX == 0 && indY == 0){
  //   for(int i = 0; i < 100; ++i){
  //     printf("%i, %f \n", i, cdf[i]);
  //   }
  // }
  // __syncthreads();
  if (indX < width && indY < height){
    int index = (width * indY + indX) * channels;
    for (int i = 0; i < channels; i++){
      image[index+i] = (unsigned char)
      (min(
        max(255.0*(cdf[image[index+i]] - cdf[0])/(1 - cdf[0]), 0.0)
      , 255.0));
    }
  }
  // __syncthreads();
  // if(indX == 0 && indY == 0){
  //   for(int i = 0; i < 100; ++i){
  //     printf("%i, %i \n", i, image[i]);
  //   }
  // }
  // __syncthreads();
}

__global__ void toFloat(unsigned char *input, float *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = (float) (input[id]/255.0);
  }
}

__global__ void char2float(unsigned char *input, float *output, int width, int height, int channels){
  int indX = blockIdx.x * blockDim.x + threadIdx.x;
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  
  // if(indX == 0){
  //   for(int i = 0; i < width * height * channels && i < 10; ++i)
  //     printf("%i, %f \n", i, input[i]);
  // }
  // __syncthreads();
  if (indX < width && indY < height){
    int index = (width * indY + indX) * channels;
    output[index] = (float)(input[index])/255.0;
    output[index + 1] = (float)(input[index + 1])/255.0;
    output[index + 2] = (float)(input[index + 2])/255.0;
    // __syncthreads();
    // if (index == 0){
    //   for(int i = 0; i < 100; ++i)
    //     printf("%i, %f \n", i, output[i]);
      // printf("%f, %f, %f\n", output[index], output[index+1],output[index+2]);
    // }
  }
  // __syncthreads();
  // if(indX == 0){
  //   for(int i = 0; i < 100; ++i)
  //     printf("%i, %i \n", i, output[i]);
  // }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImage;
  unsigned char *deviceColorImage;
  unsigned char *deviceGreyImage;
  float *deviceHistogram;
  float *deviceHistogramCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int pixelNum = imageWidth * imageHeight;
  cudaMalloc((void **) &deviceImage, pixelNum * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceColorImage, pixelNum * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGreyImage, pixelNum * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &deviceHistogramCDF, HISTOGRAM_LENGTH * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceImage, hostInputImageData, pixelNum * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(imageWidth/BLOCK_SIZE,imageHeight/BLOCK_SIZE,1);
  if (imageWidth%BLOCK_SIZE) dimGrid.x++;
  if (imageHeight%BLOCK_SIZE) dimGrid.y++;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  // float rgb to unsigned char grey
  rgb2grey<<<dimGrid, dimBlock>>>(deviceImage, deviceColorImage, deviceGreyImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // make histogram with atomicAdd
  // dim3 dimGrid3(ceil(imageWidth*imageHeight/256.0), 1, 1);
  // dim3 dimBlock3(256,1,1);
  // histogram<<<dimGrid3,dimBlock3>>>(deviceGreyImage, deviceHistogram, imageWidth*imageHeight);
  // cudaDeviceSynchronize();
  dim3 dimGrid2(pixelNum/HISTOGRAM_LENGTH,1,1);
  if (pixelNum%HISTOGRAM_LENGTH) dimGrid.x++;
  dim3 dimBlock2(HISTOGRAM_LENGTH,1,1);
  grey2histogram<<<dimGrid2, dimBlock2>>>(deviceGreyImage, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // histogram to cdf
  scan<<<1, HISTOGRAM_LENGTH/2>>>(deviceHistogram, deviceHistogramCDF);
  cudaDeviceSynchronize();
  // correct the values
  equalization<<<dimGrid, dimBlock>>>(deviceHistogramCDF, deviceColorImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // cast to float
  char2float<<<dimGrid, dimBlock>>>(deviceColorImage, deviceImage, imageWidth, imageHeight, imageChannels);




  // for(int i = 0; i < 100 && i < pixelNum * imageChannels; ++i)
  //     printf("%i, %f \n", i, deviceImage[i]);
  // for(int i = 0; i < pixelNum * imageChannels; ++i){
  //   hostOutputImageData[i] = (float)(deviceImage[i])/255.0;
  // }

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");


  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutputImageData, deviceImage, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceImage); cudaFree(deviceColorImage); cudaFree(deviceGreyImage); cudaFree(deviceHistogram); cudaFree(deviceHistogramCDF);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
