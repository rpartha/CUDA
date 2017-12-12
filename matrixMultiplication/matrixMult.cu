#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define THREADS_PER_BLK 16 //aka block size

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

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
__global__ void mtimes_gpu_sq(double *gs_a, double *gs_b, double *gs_c, int n){
    __shared__ double tile_a[THREADS_PER_BLK][THREADS_PER_BLK];
    __shared__ double tile_b[THREADS_PER_BLK][THREADS_PER_BLK];

    int r = blockIdx.y * THREADS_PER_BLK + threadIdx.y;
    int c = blockIdx.x * THREADS_PER_BLK + threadIdx.x;
    double result = 0.0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub){
        idx = r * n + sub * THREADS_PER_BLK + threadIdx.x;
        if(idx >= n*n){
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }else{
            tile_a[threadIdx.y][threadIdx.x] = gs_a[idx];
        }

        idx = (sub * THREADS_PER_BLK + threadIdx.y) * n + c;
        if(idx >= n*n){
            tile_b[threadIdx.y][threadIdx.x] = 0;
        } else{
            tile_b[threadIdx.y][threadIdx.x] = gs_b[idx];
        }
        __syncthreads(); //acts as a barrier

        for (int k = 0; k < THREADS_PER_BLK; ++k) {
            result += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads(); //acts as a barrier
    }
    if(r < n && c < n){
        gs_c[r * n + c] = result;
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

    mtimes_gpu_sq<<<dimGrid, dimBlock>>>(g_a, g_b, g_c, n);      

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

	cublasHandle_t hndl;
	cublasCreate(&hndl);

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

			cublasDdot(hndl, n, g_rv, 1, g_cv, 1, temp);
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
	cublasDestroy(hndl);

    return c; 
    
}

double *mtimes_gpu_cublas_func(double *a, double *b, int m, int n, int k){

    double *g_a, *g_b, *g_c;
    double *c = (double *) malloc(m * k * sizeof(double));

    cudaMalloc((void**)&g_a, sizeof(double) * m * n); 
    cudaMalloc((void**)&g_b, sizeof(double) * n * k); 
    cudaMalloc((void**)&g_c, sizeof(double) * m * k); 

    cublasHandle_t hndl;
    cublasCreate(&hndl);

    cublasSetMatrix(m, n, sizeof(double), a, m, g_a, m);
	cublasSetMatrix(n, k, sizeof(double), b, n, g_b, n);
    cublasSetMatrix(m, k, sizeof(double), c, m, g_c, m);
    
    double alpha = 1.0;
    double beta = 1.0;

    clock_t beg, end;
    beg = clock();
    
    cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, g_a, m, g_b, n, &beta, g_c, m); /* double precision */
    cudaThreadSynchronize();

    end = clock();
    printf("Matrix multiplication in gpu with CuBLAS gemm took: %f seconds\n", ((double)end - (double)beg) / CLOCKS_PER_SEC);

    cublasGetMatrix(m, k, sizeof(double), g_c, m, c, m);

    cudaFree(g_a);
	cudaFree(g_b);
	cudaFree(g_c);
	cublasDestroy(hndl);

    return c;
}

__global__ void mtimes_gpu_coa(double *a, double *b, double *c, int m, int n, int k){
     
   int bx = blockIdx.x;
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   __shared__ double As[THREADS_PER_BLK][THREADS_PER_BLK];

   __shared__ double Bs[THREADS_PER_BLK][THREADS_PER_BLK];

   int a0 = n * THREADS_PER_BLK * by;
   int aEnd   = a0 + n - 1;
   int a_incr  = THREADS_PER_BLK;
   int b0 = THREADS_PER_BLK * bx;
   int b_incr  = THREADS_PER_BLK * k;

   float c_sub = 0;

   for (int i = a0, j = b0; i <= aEnd; i += a_incr, j += b_incr) {
       AS(ty,tx) = a[i + n * ty + tx];
       BS(tx,ty) = b[j + k * ty + tx];

       __syncthreads();

  
       for (int k = 0; k < THREADS_PER_BLK; ++k){
            c_sub += AS(ty,k) * BS(tx,k);
       }
           
       __syncthreads();
   }

    int q = k * THREADS_PER_BLK * by + THREADS_PER_BLK * bx;
    c[q + k * ty + tx] = c_sub;
}

double *mtimes_gpu_coalescing(double *a, double *b, int m, int n, int k){

    double *g_a, *g_b, *g_c;
    double *c = (double *) malloc(m * k * sizeof(double));

    cudaMalloc((void**)&g_a, sizeof(double) * m * n); 
    cudaMalloc((void**)&g_b, sizeof(double) * n * k); 
    cudaMalloc((void**)&g_c, sizeof(double) * m * k); 

    cudaMemcpy(g_a, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, b, n * k * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 grid(k / threads.x, m / threads.y);

    clock_t beg, end;
    beg = clock();

    mtimes_gpu_coa<<< grid, threads >>>(g_a, g_b, g_c, m, n, k);

    cudaThreadSynchronize();

    end = clock();
    printf("Matrix multiplication in gpu with global coalescing took: %f seconds\n", ((double)end - (double)beg) / CLOCKS_PER_SEC);

    cudaMemcpy(c, g_c, sizeof(double) * m * k, cudaMemcpyDeviceToHost);
    cudaFree(g_a);
	cudaFree(g_b);
    cudaFree(g_c);
    
    return c;
}

__global__ void mtimes_gpu_nbc(double *a, double *b, double *c, int m, int n, int k){
    
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ double As[THREADS_PER_BLK][THREADS_PER_BLK];

  __shared__ double Bs[THREADS_PER_BLK][THREADS_PER_BLK];

  int a0 = n * THREADS_PER_BLK * by;
  int aEnd   = a0 + n - 1;
  int a_incr  = THREADS_PER_BLK;
  int b0 = THREADS_PER_BLK * bx;
  int b_incr  = THREADS_PER_BLK * k;

  float c_sub = 0;

  for (int i = a0, j = b0; i <= aEnd; i += a_incr, j += b_incr) {
      AS(ty,tx) = a[i + n * ty + tx];
      BS(ty,tx) = b[j + k * ty + tx];

      __syncthreads();

 
      for (int k = 0; k < THREADS_PER_BLK; ++k){
           c_sub += AS(ty,k) * BS(k,tx);
      }
          
      __syncthreads();
    }

   int q = k * THREADS_PER_BLK * by + THREADS_PER_BLK * bx;
   c[q + k * ty + tx] = c_sub;
}

double *mtimes_gpu_no_bank_conflicts(double *a, double *b, int m, int n, int k){
    double *g_a, *g_b, *g_c;
    double *c = (double *) malloc(m * k * sizeof(double));

    cudaMalloc((void**)&g_a, sizeof(double) * m * n); 
    cudaMalloc((void**)&g_b, sizeof(double) * n * k); 
    cudaMalloc((void**)&g_c, sizeof(double) * m * k); 

    cudaMemcpy(g_a, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, n * k * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 grid(k / threads.x, m / threads.y);

    clock_t beg, end;
    beg = clock();

    mtimes_gpu_nbc<<< grid, threads >>>(g_a, g_b, g_c, m, n, k);

    cudaThreadSynchronize();

    end = clock();
    printf("Matrix multiplication in gpu with no bank conflicts took: %f seconds\n", ((double)end - (double)beg) / CLOCKS_PER_SEC);

    cudaMemcpy(c, g_c, sizeof(double) * m * k, cudaMemcpyDeviceToHost);
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);
    
    return c;
}

