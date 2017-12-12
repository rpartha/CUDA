#pragma once

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

/* utility functions */
double *generateMatrix(int m, int n); /* randomly generated matrix */
double compareMatrices(double *a, double *b, int m, int n, int k); /* calculate MSE */
void printMatrix(double *mat, int m, int n); /* display matrix */

/* matrix multiplication cpu and gpu functions */
double *mtimes_cpu(double *a, double *b, int m, int n, int k);
double *mtimes_gpu(double *a, double *b, int m, int n, int k);
double *mtimes_gpu_cublas(double *a, double *b, int m, int n, int k);
double *mtimes_gpu_cublas_func(double *a, double *b, int m, int n, int k);
double *mtimes_gpu_opt(double *a, double *b, int m, int n, int k);

