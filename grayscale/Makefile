CP = g++
CC = gcc
NVCC = nvcc
FREEIMG_FLAG = -lfreeimage

all: grayscale_seq grayscale_opt

grayscale_seq : grayscale_seq.c 
	$(CC) -o grayscale_seq grayscale_seq.c lodepng.c

grayscale_opt : 
	$(NVCC) $(FREEIMG_FLAG) grayscale_opt.cu -o grayscale_opt

clean:
	rm grayscale_seq
	rm grayscale_opt
	rm out*.jpg
	rm out*.png
