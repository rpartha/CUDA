CC = gcc
#NVCC = nvcc
BFSC = bfs.o 

bfs : $(BFSC)
	$(CC) $(BFSC) -o bfs

bfs.o : bfs.c 
	$(CC) -c bfs.c -o bfs.o

clean:
	rm *.o
	rm bfs