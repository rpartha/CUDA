CP = g++
CC = gcc
NVCC = nvcc

all: grayscale_seq grayscale_opt

grayscale_seq : grayscale_seq.c 
	$(CC) -o grayscale_seq grayscale_seq.c lodepng.c

grayscale_opt : 
	$(CP) -c lodepng.cpp
	$(NVCC) -c grayscale_opt.cu 
	$(NVCC) -o grayscale_opt grayscale_opt.o lodepng.o

clean:
	rm *.o
	rm grayscale_seq
	rm grayscale_opt
	rm out.png
