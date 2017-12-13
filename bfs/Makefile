CC = gcc
NVCC = nvcc
BFSC = bfs.o 
BFSG = bfs_gpu.o

all: bfs bfs_gpu

bfs_gpu : $(BFSG)
	$(NVCC) $(BFSG) -o bfs_gpu

bfs : $(BFSC)
	$(CC) $(BFSC) -o bfs

bfs_gpu.o: bfs_gpu.cu
	$(NVCC) -c bfs_gpu.cu -o bfs_gpu.o

bfs.o : bfs.c 
	$(CC) -c bfs.c -o bfs.o

clean:
	rm *.o
	rm bfs
	rm bfs_gpu