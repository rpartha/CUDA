#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#define CHAgridSizegridSizeELS 3
#define THREADS_PER_BLK 16

typedef struct{
    unsigned width;
    unsigned height;
    unsigned char* image;
}RGB_IMG;

__global__ void rgb2gray_helper(unsigned char* image_in, unsigned width, unsigned height){
    
    float gray;
    float r, g, b;

	int rows = blockIdx.x*blockDim.x + threadIdx.x;
	int cols = blockIdx.y*blockDim.y + threadIdx.y;

	if (rows < width && cols < height) {
		r = image_in[4*width*cols + 4*rows + 0];
		g = image_in[4*width*cols + 4*rows + 1];
        b = image_in[4*width*cols + 4*rows + 2];
        
        gray = .299f*r + .587f*g + .114f*b;
        
		image_in[4*width*cols + 4*rows + 0] = gray;
		image_in[4*width*cols + 4*rows + 1] = gray;
		image_in[4*width*cols + 4*rows + 2] = gray;
		image_in[4*width*cols + 4*rows + 3] = 255;
	}
}

void convertRGBtoGrayscaleOpt(RGB_IMG* in){
    
	unsigned char* image_in = in->image;
    unsigned char* deviceInputImage;

    unsigned width = in->width;
    unsigned height = in->height;

    int gridSize = (int)width*(int)height; 

    size_t size = gridSize*4*sizeof(unsigned char);
	
    
	int device_count = 0;
	cudaError_t status = cudaGetDeviceCount(&device_count);
	
	status = cudaMalloc((void **) &deviceInputImage, size);
	
	cudaMemcpy(deviceInputImage, image_in,  size, cudaMemcpyHostToDevice);
	
    cudaEvent_t beg, end;
    float totalTime;

    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudaEventRecord(beg, 0);

	dim3 dimThread(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 dimGrid(in->width / dimThread.x, in->height / dimThread.y);
    
    rgb2gray_helper<<<dimGrid, dimThread>>>(deviceInputImage, in->width, in->height);
    
	cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&totalTime, beg, end);
    printf("The total time taken: %f\n", totalTime);
	
	cudaMemcpy(image_in, deviceInputImage, size, cudaMemcpyDeviceToHost);
	
	cudaFree(deviceInputImage);
}

void decode(RGB_IMG* in, const char* fname){
    
    unsigned char* img;
    size_t img_size;
    
    lodepng_load_file(&img, &img_size, fname);

    unsigned err = lodepng_decode32(&in->image, &in->width, &in->height, img, img_size);

    if(err){
        printf("There was an error %u: %s\n", err, lodepng_error_text(err));
    }
}

void encode(RGB_IMG* in, const char* fname){
   unsigned err = lodepng_encode32_file(fname, in->image, in->width, in->height);

   if(err){
       printf("There was an error %u: %s\n", err, lodepng_error_text(err));
   }
}

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Usage: ./<png image>");
        exit(1);
    }

    else{
       
       const char* fname = argv[1];
       RGB_IMG in;

       decode(&in, fname);
       convertRGBtoGrayscaleOpt(&in);
       encode(&in, "out.png");

       return 0; 
    }
}