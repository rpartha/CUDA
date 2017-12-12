#include "matrix.h"

double *generateMatrix(int m, int n){
    double *mat = (double *) malloc(m * n * sizeof(double));
	srand(time(NULL));

	int i;
	for (i = 0; i < (m * n); i++) {
		mat[i] = (double)rand() / (double)RAND_MAX * 1000;
    }
    
    return mat;  
}

void printMatrix(double *mat, int m, int n){
	int i;
	for (i = (m*n)-25; i < (m * n); i++) {
		printf("%f\n", mat[i]);
    }
}

double compareMatrices(double *a, double *b, int m, int n, int k){
	double mean_sq_error = 0;

	int i;
	for(i = 0; i < m * n; i++){
		mean_sq_error = (a[i] - b[i]) * (a[i] - b[i]);
	}

	mean_sq_error /= ((double) m * (double) n);

	return mean_sq_error;
}

double *mtimes_cpu(double *a, double *b, int m, int n, int k){
	double *c = (double *) calloc(m*k, sizeof(double));

	for (int i = 0; i < m; i++) {
        for	 (int j = 0; j < k; j++) {
            for (int h = 0; h < n; h++) {
                c[(m * i) + j] += a[(i * n) + h] * b[(h * k) + j];
            }
        }            
	}

	return c;
}

int main(int argc, char *argv[]){
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);

	double *a = generateMatrix(m, n);
	double *b = generateMatrix(n, k);

	double *c_cpu = mtimes_cpu(a, b, m, n, k);
	double *c_gpu = mtimes_gpu(a, b, m, n, k);	
	double *cc_gpu = mtimes_gpu_cublas(a, b, m, n, k);
	double *ccg_gpu = mtimes_gpu_cublas_func(a, b, m, n, k);
	//double *c_gpu_opt = mtimes_gpu_opt(a, b, m, n, k);
	
	double res1 = compareMatrices(c_cpu, c_gpu, m, n, k);
	printf("GPU MSE: %f\n", res1);

	double res2 = compareMatrices(c_cpu, cc_gpu, m, n, k);
	printf("GPU CuBLAS MSE: %f\n", res2);

	double res3 = compareMatrices(c_cpu, ccg_gpu, m, n, k);
	printf("GPU CuBLAS gemm MSE: %f\n", res3);
}