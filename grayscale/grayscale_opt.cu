#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#define CHANNELS 3
#define THREADS_PER_BLK 32

#define max3(a,b,c) a > b ? (a > c ? a : c) : (b > c ? b : c)
#define min(a,b) a < b ? a : b
#define min3(a,b,c) min(a, min(b,c))

typedef struct{
    unsigned width;
    unsigned height;
    unsigned char* image;
    int option;
}RGB_IMG;

__global__ void rgb2gray_helper(unsigned char* image_in, unsigned width, unsigned height, int option){
    
    float gray;
    float r, g, b;

	int rows = blockIdx.x*blockDim.x + threadIdx.x;
	int cols = blockIdx.y*blockDim.y + threadIdx.y;

	if (rows < width && cols < height) {

        if(option == 0){
            r = image_in[4*width*cols + 4*rows];
            g = image_in[4*width*cols + 4*rows + 1];
            b = image_in[4*width*cols + 4*rows + 2];
            
            gray = .299f*r + .587f*g + .114f*b;
    
            //gray = (max3(r, g, b) + min3(r,g,b))/2.0;
            
            image_in[4*width*cols + 4*rows] = gray;
            image_in[4*width*cols + 4*rows + 1] = gray;
            image_in[4*width*cols + 4*rows + 2] = gray;
            image_in[4*width*cols + 4*rows + 3] = 255;
        } 

        else if(option == 1){
            r = image_in[4*width*cols + 4*rows];
            g = image_in[4*width*cols + 4*rows + 1];
            b = image_in[4*width*cols + 4*rows + 2];
            
            gray = .2126f*r + .7152f*g + .0722f*b;
    
            //gray = (max3(r, g, b) + min3(r,g,b))/2.0;
            
            image_in[4*width*cols + 4*rows] = gray;
            image_in[4*width*cols + 4*rows + 1] = gray;
            image_in[4*width*cols + 4*rows + 2] = gray;
            image_in[4*width*cols + 4*rows + 3] = 255;
        }

        else if(option == 2){
            r = image_in[4*width*cols + 4*rows];
            g = image_in[4*width*cols + 4*rows + 1];
            b = image_in[4*width*cols + 4*rows + 2];
            
            //gray = .2126f*r + .7152f*g + .0722f*b;
    
            gray = (max3(r, g, b) + min3(r,g,b))/2.0;
            
            image_in[4*width*cols + 4*rows] = gray;
            image_in[4*width*cols + 4*rows + 1] = gray;
            image_in[4*width*cols + 4*rows + 2] = gray;
            image_in[4*width*cols + 4*rows + 3] = 255;
        }

        else{
            r = image_in[4*width*cols + 4*rows];
            g = image_in[4*width*cols + 4*rows + 1];
            b = image_in[4*width*cols + 4*rows + 2];
            
            gray = .299f*r + .587f*g + .114f*b;
    
            //gray = (max3(r, g, b) + min3(r,g,b))/2.0;
            
            image_in[4*width*cols + 4*rows] = gray;
            image_in[4*width*cols + 4*rows + 1] = gray;
            image_in[4*width*cols + 4*rows + 2] = gray;
            image_in[4*width*cols + 4*rows + 3] = 255;
        }
		
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
    
    rgb2gray_helper<<<dimGrid, dimThread>>>(deviceInputImage, in->width, in->height, in->option);
    
	cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&totalTime, beg, end);
    printf("The total time taken: %f ms\n", totalTime);
	
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
    if(argc != 4){
        printf("Usage: ./<input image> <output image> <option>");
        exit(1);
    }

    else{
       
       const char* fname = argv[1];
       RGB_IMG in;

       in.option = atoi(argv[3]);

       decode(&in, fname);
       convertRGBtoGrayscaleOpt(&in);
       encode(&in, argv[2]);

       return 0; 
    }
}