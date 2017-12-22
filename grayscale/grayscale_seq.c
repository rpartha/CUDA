#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHANNELS 3

void rgb2gray_cpu(char* fname_in, char* fname_out){
    unsigned char* img_in;
    unsigned char* img_out;

    unsigned width;
    unsigned height;

    unsigned err = lodepng_decode32_file(&img_in, &width, &height, fname_in); // decode 32-bit raw RGBA input image

    if(err){
        printf("error %u: %s\n", err, lodepng_error_text(err));
    }

    img_out = malloc(sizeof(unsigned char)*width*height*4);

    /* compute grayscale by averaging pixels */
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int psum = 0;
            int avg = 0;
            for(int k = 0; k < CHANNELS; k++){
                psum += img_in[4*width*i + 4*j + k];
                avg = psum/3;
                for(int l = 0; l < CHANNELS; l++){
                    img_out[4*width*i + 4*j + l] = avg;
                }
            }
            img_out[4*width*i + 4*j + 3] = img_in[4*width*i + 4*j + 3]; 
        }
    }

    lodepng_encode32_file(fname_out, img_out, width, height); //encode 32-bit raw output image

    free(img_in);
    free(img_out);
}

int main(int argc, char* argv[]){
    if(argc != 3){
        printf("Usage: ./grayscale_seq <input file> <output file>");
        exit(0);
    }

    else{
        char* fname_in = argv[1];
        char* fname_out = argv[2];

        clock_t start, end;
        double cpu_time_used;

        start = clock();
        rgb2gray_cpu(fname_in, fname_out);
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("CPU version took: %f seconds\n", cpu_time_used);
        
    }
}

