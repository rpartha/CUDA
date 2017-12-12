CC = g++
NVCC = nvcc
OBJS = matrixMult.o matrix.o 
CUBLAS_FLAG = -lcublas

mm : $(OBJS)
	$(NVCC) $(CUBLAS_FLAG) $(OBJS) -o mm

matrixMult.o : matrixMult.cu matrix.h
	$(NVCC) $(CUBLAS_FLAG) -c matrixMult.cu -o matrixMult.o

matrix.o : matrix.c matrix.h
	$(CC) -c matrix.c -o matrix.o

clean:
	rm *.o
	rm mm