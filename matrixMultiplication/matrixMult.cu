#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define THREADS_PER_BLK 16 //aka block size

__global__ void mtimes(double *g_a, double *g_b, double *g_c, int m, int n, int k){ 
    int r = blockIdx.y * blockDim.y + threadIdx.y; 
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0; //thread-local variable
    if( c < k && r < m) {
        for(int i = 0; i < n; i++){
            sum += g_a[r * n + i] * g_b[i * k + c];
        }
        g_c[r * k + c] = sum; //write sum to global memory
    }                       
} 

/* square matrices using tiling*/
__global__ void mtimes_sq(double *gs_a, double *gs_b, double *gs_c, int m, int n, int k){
	__shared__ double s_a[THREADS_PER_BLK][THREADS_PER_BLK];
	__shared__ double s_b[THREADS_PER_BLK][THREADS_PER_BLK];

	unsigned int r = THREADS_PER_BLK * blockIdx.y + threadIdx.y;
    unsigned int c = THREADS_PER_BLK * blockIdx.x + threadIdx.x;
    
	unsigned int i, j;

	double x = 0.0;
	for (i = 0; i < (THREADS_PER_BLK + n - 1) / THREADS_PER_BLK; i++){
		if ((i * THREADS_PER_BLK + threadIdx.x < n) && (r < m))
		{
			s_a[threadIdx.y][threadIdx.x] = gs_a[(r * n) + (i * THREADS_PER_BLK) + threadIdx.x];
		} 
		else 
		{
			s_a[threadIdx.y][threadIdx.x] = 0.0;
		}

		if ((i * THREADS_PER_BLK + threadIdx.y < n) && (c < k)){
			s_b[threadIdx.y][threadIdx.x] = gs_b[c + k * (i * THREADS_PER_BLK + threadIdx.y)];
		}
		else{
			s_b[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads();

		for (j = 0; j < THREADS_PER_BLK; j++){
			x += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
		}
		__syncthreads();
	}

	if ((r < m) && (c < k)){
		gs_c[(blockIdx.y * blockDim.y + threadIdx.y) * k + (blockIdx.x * blockDim.x) + threadIdx.x] = x;
    }
}

double *mtimes_gpu(double *a, double *b, int m, int n, int k){
    
    double *g_a, *g_b, *g_c;
    double *c = (double *) malloc(m * k * sizeof(double));

    cudaMalloc((void**)&g_a, sizeof(double) * m * n); 
    cudaMalloc((void**)&g_b, sizeof(double) * n * k); 
    cudaMalloc((void**)&g_c, sizeof(double) * m * k); 

    cudaMemcpy(g_a, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, b, n * k * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 dimGrid((int)ceil((double)m / (double)dimBlock.x), (int)ceil((double)k / (double)dimBlock.y));

    clock_t beg, end;
    beg = clock();

    mtimes_sq<<<dimGrid, dimBlock>>>(g_a, g_b, g_c, m, n, k);    

    cudaThreadSynchronize();

    end = clock();
    printf("Matrix multiplication in gpu (initial) took: %f secs\n", ((double)end - (double)beg) / CLOCKS_PER_SEC);

    cudaMemcpy(c, g_c, sizeof(double) * m * k, cudaMemcpyDeviceToHost);
	cudaFree(g_a);
	cudaFree(g_b);
	cudaFree(g_c);

    return c;
}

double *mtimes_gpu_cublas(double *a, double *b, int m, int n, int k){

	double *g_rv, *g_cv, *g_c;

	double *temp = (double *) malloc(sizeof(double));
    double *c = (double *) malloc(m * k * sizeof(double));
    
	cudaMalloc((void**)&g_rv, n * sizeof(double));
	cudaMalloc((void**)&g_cv, n * sizeof(double));
	cudaMalloc((void**)&g_c, sizeof(double));

	cublasHandle_t handle;
	cublasCreate(&handle);

	int i, j, h;
	double *rv = (double *) malloc(n * sizeof(double));
	double *cv = (double *) malloc(n * sizeof(double));

	clock_t beg, end;
	double tot = 0;

	for (i = 0; i < m; i++){
		for (j = 0; j < k; j++){
			for (h = 0; h < n; h++){
				rv[h] = a[(i * n) + h];
				cv[h] = b[(h * k) + j];
			}
			cublasSetVector(n, sizeof(double), rv, 1, g_rv, 1);
			cublasSetVector(n, sizeof(double), cv, 1, g_cv, 1);
			cublasGetVector(n, sizeof(double), g_rv, 1, rv, 1);
			cublasGetVector(n, sizeof(double), g_cv, 1, cv, 1);

			beg = clock();

			cublasDdot(handle, n, g_rv, 1, g_cv, 1, temp);
			cudaThreadSynchronize();

			end = clock();

			c[(i * k) + j] = *temp;
			
			tot += ((double)end - (double)beg);
		}
	}


	printf("Matrix Multiplication with CuBLAS Library took: %f seconds\n", tot / CLOCKS_PER_SEC);

	free(temp);
	cudaFree(g_rv);
	cudaFree(g_cv);
	cublasDestroy(handle);

    return c; 
    
}

double *mtimes_gpu_cublas_func(double *a, double *b, int m, int n, int k){

    double *g_a, *g_b, *g_c;
    double *c = (double *) malloc(m * k * sizeof(double));

    cudaMalloc((void**)&g_a, sizeof(double) * m * n); 
    cudaMalloc((void**)&g_b, sizeof(double) * n * k); 
    cudaMalloc((void**)&g_c, sizeof(double) * m * k); 

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(m, n, sizeof(double), a, m, g_a, m);
	cublasSetMatrix(n, k, sizeof(double), b, n, g_b, n);
    cublasSetMatrix(m, k, sizeof(double), c, m, g_c, m);
    
    double alpha = 1.0;
    double beta = 1.0;

    clock_t beg, end;
    beg = clock();
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, g_a, m, g_b, n, &beta, g_c, m);
    cudaThreadSynchronize();

    end = clock();
    printf("Matrix multiplication in gpu with CuBLAS gemm took: %f seconds\n", ((double)end - (double)beg) / CLOCKS_PER_SEC);

    cublasGetMatrix(m, k, sizeof(double), g_c, m, c, m);

    cudaFree(g_a);
	cudaFree(g_b);
	cudaFree(g_c);
	cublasDestroy(handle);

    return c;
}

