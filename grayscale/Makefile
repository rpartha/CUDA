CC = g++
NVCC = nvcc
FREEIMG_FLAG = -lfreeimage

all: grayscale grayscaleopt

grayscale : 
	$(NVCC) $(FREEIMG_FLAG) rgb2grayscale.cu -o grayscale

grayscaleopt : 
	$(CC) -c lodepng.cpp
	$(NVCC) -c rgb2grayscaleopt.cu 
	$(NVCC) -o grayscaleopt rgb2grayscaleopt.o lodepng.o

clean:
	rm *.o
	rm grayscale
	rm grayscaleopt
	rm out.png
	rm out.jpg
