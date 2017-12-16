#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <FreeImage.h>

using namespace std;

#define CHANNELS 3
#define THREADS_PER_BLK 32

struct RGB{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

__global__ void convertRGBToGrayscale(RGB* in, float* out, int rows, int cols){
    if (blockIdx.x == (int) cols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= cols) return;
    else if (blockIdx.y == (int) rows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= rows) return;
    
    unsigned long threadOffset = threadIdx.x + threadIdx.y * cols;
    unsigned long blockOffset = blockIdx.y * blockDim.x * cols + blockDim.y * blockIdx.x;

    out[(unsigned long)(threadOffset + blockOffset)] = float(in[(unsigned long)(threadOffset + blockOffset)].r) * 0.2989f + float(in[(unsigned long)(threadOffset + blockOffset)].g) * 0.587f + float(in[threadOffset + blockOffset].b) * 0.114f;
}

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Usage: ./<image>");
        exit(1);
    }

    else{
        FreeImage_Initialise(); 
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(argv[1]);
        FIBITMAP* fibmap = FreeImage_Load(format, argv[1]);
    
        int m = FreeImage_GetHeight(fibmap);
        int n = FreeImage_GetWidth(fibmap);
        int pitch = FreeImage_GetPitch(fibmap);
    
        RGB* hostInput = new RGB[m * n];
        float* hostOutput = new float[m * n];

        RGB* deviceInput;
        float* deviceOutput;
        
        cudaMalloc((void **) &deviceInput, sizeof(RGB)*m*n);
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
        
        cudaMemcpy(deviceInput, hostInput, sizeof(RGB)*m*n, cudaMemcpyHostToDevice);
        cudaEvent_t beg, end;
        float totalTime;

        cudaEventCreate(&beg);
        cudaEventCreate(&end);
    
        cudaEventRecord(beg, 0);

        dim3 dimGrid(ceil(n/THREADS_PER_BLK), ceil(m/THREADS_PER_BLK), 1);
        dim3 dimThread(THREADS_PER_BLK, THREADS_PER_BLK, 1);

        convertRGBToGrayscale<<<dimGrid,dimThread>>>(deviceInput, deviceOutput, m, n);

        cudaPeekAtLastError();

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
    
        cudaEventElapsedTime(&totalTime, beg, end);
        printf("The total time taken: %f\n", totalTime);

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
    
        FreeImage_Save(FIF_JPEG, fibmap, "out.jpeg", JPEG_DEFAULT);
        FreeImage_DeInitialise();
    
        return 0; 
    }
}