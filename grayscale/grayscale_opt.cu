#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <FreeImage.h>

using namespace std;

#define CHANNELS 3
#define THREADS_PER_BLK 32

#define max3(a,b,c) a > b ? (a > c ? a : c) : (b > c ? b : c)
#define min(a,b) a < b ? a : b
#define min3(a,b,c) min(a, min(b,c))

struct RGB_IMG{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

__global__ void convertRGBToGrayscale(RGB_IMG* in, float* out, int rows, int cols, int option){
    if (blockIdx.x == (int) cols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= cols) return;
    else if (blockIdx.y == (int) rows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= rows) return;
    
    unsigned long threadOffset = threadIdx.x + threadIdx.y * cols;
    unsigned long blockOffset = blockIdx.y * blockDim.x * cols + blockDim.y * blockIdx.x;

    float red, green, blue;

    red = in[(unsigned long)(threadOffset + blockOffset)].r;
    green = in[(unsigned long)(threadOffset + blockOffset)].g;
    blue = in[(unsigned long)(threadOffset + blockOffset)].b;

    if(option == 0){
        out[(unsigned long)(threadOffset + blockOffset)] = float(red) * 0.299f + float(green) * 0.587f + float(blue) * 0.114f;
    } 

    else if(option == 1){
        out[(unsigned long)(threadOffset + blockOffset)] = float(red) * 0.2126f + float(green) * 0.7152f + float(blue) * 0.0722f;

    }
    
    else if(option == 3){
        out[(unsigned long)(threadOffset + blockOffset)] = (float(red) + float(green)  + float(blue))/3.0;
    }
    
    else if(option == 4){
        out[(unsigned long)(threadOffset + blockOffset)] = (max3(red, green, blue) + min3(red, green, blue))/2.0;
    }

    else if(option == 5){
        out[(unsigned long)(threadOffset + blockOffset)] = max3(red, green, blue);
    }

    else if(option == 6){
        out[(unsigned long)(threadOffset + blockOffset)] = min3(red, green, blue);
    }

    else{
        out[(unsigned long)(threadOffset + blockOffset)] = (float(red) + float(green)  + float(blue))/3.0;
    }
}

int main(int argc, char* argv[]){
    if(argc != 4){
        printf("Usage: ./<input image> <output image file> <option>");
        exit(1);
    }

    else{
        FreeImage_Initialise(); 
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(argv[1]);
        FIBITMAP* fibmap = FreeImage_Load(format, argv[1]);
    
        int m = FreeImage_GetHeight(fibmap);
        int n = FreeImage_GetWidth(fibmap);
        int pitch = FreeImage_GetPitch(fibmap);
    
        RGB_IMG* hostInput = new RGB_IMG[m * n];
        float* hostOutput = new float[m * n];

        RGB_IMG* deviceInput;
        float* deviceOutput;
        
        cudaMalloc((void **) &deviceInput, sizeof(RGB_IMG)*m*n);
        cudaMalloc((void **) &deviceOutput, sizeof(float)*m*n);
    
        FREE_IMAGE_TYPE freeImageType = FreeImage_GetImageType(fibmap);
        int k = 0;
        if(freeImageType == FIT_BITMAP) {
            BYTE* bits = (BYTE*)FreeImage_GetBits(fibmap);
            for(int i = 0; i < m; i++) {
                BYTE* px = (BYTE *) bits;
                for(int j =  0; j < n; j++) {
                    hostInput[k].r = px[FI_RGBA_RED];
                    hostInput[k].g = px[FI_RGBA_GREEN];
                    hostInput[k++].b = px[FI_RGBA_BLUE];
                    px += 3;
                }
                bits += pitch;
            }
        }
        
        cudaMemcpy(deviceInput, hostInput, sizeof(RGB_IMG)*m*n, cudaMemcpyHostToDevice);
        cudaEvent_t beg, end;
        float totalTime;

        cudaEventCreate(&beg);
        cudaEventCreate(&end);
    
        cudaEventRecord(beg, 0);

        dim3 dimGrid(ceil(n/THREADS_PER_BLK), ceil(m/THREADS_PER_BLK), 1);
        dim3 dimThread(THREADS_PER_BLK, THREADS_PER_BLK, 1);

        convertRGBToGrayscale<<<dimGrid,dimThread>>>(deviceInput, deviceOutput, m, n, atoi(argv[3]));

        cudaPeekAtLastError();

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
    
        cudaEventElapsedTime(&totalTime, beg, end);
        printf("The total time taken: %f ms\n", totalTime);

        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    
        cudaMemcpy(hostOutput, deviceOutput, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        cudaFree(deviceOutput);
        cudaFree(deviceInput);
        cudaDeviceSynchronize();
    
        BYTE* bits = (BYTE*)FreeImage_GetBits(fibmap);
        k = 0;
        if(freeImageType == FIT_BITMAP) {
            for(int i = 0; i < m; i++) {
                BYTE* px = (BYTE *) bits;
                for(int j =  0; j < n; j++) {
                    px[FI_RGBA_RED] = px[FI_RGBA_GREEN] = px[FI_RGBA_BLUE] = hostOutput[k++];
                    px += 3;
                }
                bits += pitch;
            }
        }
    
        FreeImage_Save(FIF_JPEG, fibmap, argv[2], JPEG_DEFAULT);
        FreeImage_DeInitialise();
    
        return 0; 
    }
}