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

__global__ void grey2histogram(unsigned char *inputImagePtr, float *histPtr, int width, int height){
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  
  atomicAdd( &(histPtr[inputImagePtr[indX]]), 1);
  __syncthreads();
  
  if (indX<256) {
    histPtr[indX] /= (float)(width*height);
  }
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
}

__global__ void equalization(unsigned char *imagePtr, float *histCdfPtr, int 
width, int height, int channels){
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  if (indX < width && indY < height){
    int index = (width * indY + indX) * channels;
    for (int i=0; i<3; i++){
      // if (index == 0)
      // printf("i: %d, pix: %d, cdf: %f, value: %f, result: %d\n", index+i, imagePtr[index+i],histCdfPtr[imagePtr[index+i]], 255.0*histCdfPtr[imagePtr[index+i]], (unsigned char)(min(max(255.0*(histCdfPtr[imagePtr[index+i]] - histCdfPtr[0])/(1 - histCdfPtr[0]), 0.0), 255.0)));
      
      imagePtr[index+i] = (unsigned char)
      (min(max(255.0*(histCdfPtr[imagePtr[index+i]] - histCdfPtr[0])/(1 - histCdfPtr[0]), 0.0), 255.0));
    }
  }
}
__global__ void char2float(unsigned char *inputImagePtr, float *outputImagePtr, int width, int height, int channels){
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  if (indX < width && indY < height){
    int index = (width * indY + indX) * channels;
    outputImagePtr[index] = (float)(inputImagePtr[index])/255.0;
    outputImagePtr[index + 1] = (float)(inputImagePtr[index + 1])/255.0;
    outputImagePtr[index + 2] = (float)(inputImagePtr[index + 2])/255.0;
    // if (index == 0) printf("%f, %f, %f\n",outputImagePtr[index],outputImagePtr[index+1],outputImagePtr[index+2]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  // float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
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
  wbTime_stop(Generic, "Importing data and creating memory on host");
  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  
  // allocate memory in cuda device
  cudaMalloc((void **) &deviceInputImage, imageWidth * imageHeight * 
  imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceColorImage, imageWidth * imageHeight * 
  imageChannels * sizeof(unsigned char));
  and print an error code
  cudaMalloc((void **) &deviceGreyImage, imageWidth * imageHeight * 
  sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * 
  sizeof(float));
  cudaMalloc((void **) &deviceHistogramCDF, HISTOGRAM_LENGTH * 
  sizeof(float));
  
  //copy input image to cuda device
  cudaMemcpy(deviceInputImage, hostInputImageData,imageWidth * 
  imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 dimGrid(imageWidth/BLOCK_SIZE,imageHeight/BLOCK_SIZE,1);
  if (imageWidth%BLOCK_SIZE) dimGrid.x++;
  if (imageHeight%BLOCK_SIZE) dimGrid.y++;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  rgb2grey<<<dimGrid, dimBlock>>>(deviceImage, deviceColorImage, deviceGreyImage, imageWidth, imageHeight, imageChannels);
  grey2histogram<<<imageHeight*imageWidth/256, 256>>>(deviceGreyImage, 
  deviceHistogram, imageWidth, imageHeight);
  scan<<<1, HISTOGRAM_LENGTH/2>>>(deviceHistogram, deviceHistogramCDF);
  equalization<<<dimGrid, dimBlock>>>(deviceColorImage, deviceHistogramCDF, 
  imageWidth, imageHeight, imageChannels);
  char2float<<<dimGrid, dimBlock>>>(deviceColorImage, deviceInputImage, 
  imageWidth, imageHeight, imageChannels);
  
  cudaDeviceSynchronize();
  cudaMemcpy(hostInputImageData, deviceInputImage, imageWidth * 
  imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage,hostInputImageData);
  
  wbSolution(args, outputImage);
  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(deviceColorImage);
  cudaFree(deviceGreyImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceHistogramCDF);


  return 0;
}
