#include <wb.h>





#define HISTOGRAM_LENGTH 256
#define SCAN (2 * HISTOGRAM_LENGTH)

#define uint8_t unsigned char
#define uint_t unsigned int

#define gd gridDim.x
#define bd blockDim.x
#define bx blockIdx.x
#define tx threadIdx.x

//@@ insert code here
//@@ Kernels

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

__global__ void CalcHist(uint8_t* imgIn, uint_t* histOut, int imgSize){
  __shared__ uint_t hist[HISTOGRAM_LENGTH];

  int idx = bd * bx + tx;

  if(tx < HISTOGRAM_LENGTH) hist[tx] = 0;
  __syncthreads();

  if(idx < imgSize) atomicAdd(&hist[imgIn[idx]], 1);
  __syncthreads();

  if(tx < HISTOGRAM_LENGTH){
    atomicAdd(&histOut[tx], hist[tx]);
  }

}



__global__ void HistToCDF(uint_t* histIn, float* CDFOut, int imgSize){

  __shared__ float temp[SCAN];

  int idx = bx * bd + tx;

  if(idx < HISTOGRAM_LENGTH){
    temp[tx] = histIn[idx];
  }

  unsigned int s;
  for(s = 1; s <= HISTOGRAM_LENGTH; s = s * 2){
    __syncthreads();
    unsigned int j = (2 * s * (1 + tx)) - 1;
    if(j < HISTOGRAM_LENGTH && j < SCAN){
      temp[j] = temp[j] + temp[j - s];
    }
  }

  for(s = HISTOGRAM_LENGTH / 2; s > 0; s = s / 2){
    __syncthreads();
    unsigned int j = (2 * s * (1 + tx)) - 1;
    if(j + s < HISTOGRAM_LENGTH && j + s < SCAN){
      temp[j + s] = temp[j + s] + temp[j];
    }
  }
  __syncthreads();

  if(idx < HISTOGRAM_LENGTH){
    CDFOut[idx] = temp[tx] / imgSize;
  }
}

__global__ void EqualizeByCDF(uint8_t* imgIn, float* imgOut, float* CDFIn, int imgSize){
  int idx = bx * bd + tx;

  if(idx < imgSize){
    float temp = 255 * (CDFIn[imgIn[idx]] - CDFIn[0]) / (1 - CDFIn[0]) / (HISTOGRAM_LENGTH - 1);

    imgOut[idx] = (float) min(max(temp, 0.0), 255.0);
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
  float *hostOutputImageData;
  const char *inputImageFile;

  float *deviceIn;
  uint8_t *deviceUChar;
  uint8_t *deviceGray;
  uint_t *deviceHist;
  float *deviceCDF;
  float *deviceOut;



  //@@ Insert more code here

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

  //@@ get the correct image pointers
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ allocate memory
  cudaMalloc((void**) &deviceIn, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceUChar, imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
  cudaMalloc((void**) &deviceGray, imageWidth * imageHeight * sizeof(uint8_t));
  cudaMalloc((void**) &deviceHist, HISTOGRAM_LENGTH * sizeof(uint_t));
  cudaMalloc((void**) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**) &deviceOut, imageWidth * imageHeight * imageChannels * sizeof(float));


  //@@ copy to gpu and set
  cudaMemset((void *) deviceHist, 0, HISTOGRAM_LENGTH * sizeof(uint_t));
  cudaMemset((void *) deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(deviceIn, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);


  //@@ grid block dims
  dim3 block(HISTOGRAM_LENGTH);
  dim3 grid(((imageWidth*imageHeight*imageChannels) - 1) / HISTOGRAM_LENGTH + 1);



  //@@ launch each kernel
  rgb2grey<<<dimGrid, dimBlock>>>(deviceImage, deviceUChar, deviceGray, imageWidth, imageHeight, imageChannels);
  CalcHist<<<grid, block>>>(deviceGray, deviceHist, imageWidth * imageHeight);
  HistToCDF<<<grid, block>>>(deviceHist, deviceCDF, imageWidth * imageHeight);
  EqualizeByCDF<<<grid, block>>>(deviceUChar, deviceOut, deviceCDF, imageWidth * imageHeight * imageChannels);

  //@@ copy from gpu
  cudaMemcpy(hostOutputImageData, deviceOut, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  //@@ set the output image
  wbImage_setData(outputImage, hostOutputImageData);

  wbSolution(args, outputImage);

  //@@ insert code here
  //@@ free memory
  cudaFree(deviceIn);
  cudaFree(deviceUChar);
  cudaFree(deviceGray);
  cudaFree(deviceHist);
  cudaFree(deviceCDF);
  cudaFree(deviceOut);


  return 0;
}