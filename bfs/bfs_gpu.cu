#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

typedef struct Vertex {
	int data;
	int visited;
    int xpos;
    int ypos;
} Vertex;

typedef struct Node {
	int data;
	int* pos; 
	struct Node* next;
} Node;

void transfer1(Vertex** g, Vertex** arr, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            (*arr)[j + i*size] = g[i][j];
            (*arr)[j + i*size].xpos = i;
            (*arr)[j + i*size].ypos = j;
        }
    }
}
    
void transfer2(Vertex*** graph, Vertex* arr, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            (*graph)[i][j].visited = arr[j + i*size].visited;
        }
    }
}

void display(Vertex** graph, int size) {
	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (graph[i][j].data != -1) {
				if (graph[i][j].data < 10) {
					printf("%d  ", graph[i][j].data);
				}
				else {
					printf("%d ", graph[i][j].data);
				}
			}
			else {
				printf("   ");
			}
		}
		printf("\n");
	}

	printf("\n");

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (graph[i][j].data != -1) {
				printf("%d ", graph[i][j].visited);
			}
			else {
				printf("  ");
			}
		}
		printf("\n");
	}
}

void freeGraph(Vertex** graph, int size) {
	for(int i = 0; i < size; ++i) {
		free(graph[i]);
	}

	free(graph);
}

Vertex** build(Vertex** g, int size) { 
	Vertex** graph;
	graph = g;
	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (i == 0) { 
				if (j == 0) {
					if (graph[i+1][j].data == -1 && graph[i][j+1].data == -1 && graph[i+1][j+1].data == -1) {
						graph[i+1][j].data = rand() % 100; 
						graph[i][j+1].data = rand() % 100; 
						graph[i+1][j+1].data = rand() % 100; 
					}
				}
				else if (j == size - 1) { 
					if (graph[i+1][j].data == -1 && graph[i][j-1].data == -1 && graph[i+1][j-1].data == -1) {
						graph[i+1][j].data = rand() % 100;
						graph[i][j-1].data = rand() % 100;
						graph[i+1][j-1].data = rand() % 100;
					}
				}
				else { 
					if (graph[i][j-1].data == -1 && graph[i][j+1].data == -1 &&
						graph[i+1][j-1].data == -1 && graph[i+1][j].data == -1 && graph[i+1][j+1].data == -1) {

						graph[i][j-1].data = rand() % 100;
						graph[i][j+1].data = rand() % 100;
						graph[i+1][j-1].data = rand() % 100;
						graph[i+1][j].data = rand() % 100;
						graph[i+1][j+1].data = rand() % 100;

					}
				}
			}
			else if (i == size - 1) {
				if (j == 0) { 
					if (graph[i-1][j].data == -1 && graph[i][j+1].data == -1 && graph[i-1][j+1].data == -1) {
						graph[i-1][j].data = rand() % 100; 
						graph[i][j+1].data = rand() % 100; 
						graph[i-1][j+1].data = rand() % 100;
					}
				}
				else if (j == size - 1) { 
					if (graph[i-1][j].data == -1 && graph[i][j-1].data == -1 && graph[i-1][j-1].data == -1) {
						graph[i-1][j].data = rand() % 100;
						graph[i][j-1].data = rand() % 100;
						graph[i-1][j-1].data = rand() % 100;
					}
				}
				else { 
					if (graph[i][j-1].data == -1 && graph[i][j+1].data == -1 &&
						graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 && graph[i-1][j+1].data == -1) {

						graph[i][j-1].data = rand() % 100;
						graph[i][j+1].data = rand() % 100;
						graph[i-1][j-1].data = rand() % 100;
						graph[i-1][j].data = rand() % 100;
						graph[i-1][j+1].data = rand() % 100;

					}
				}
			}
			else {
				if (j == 0) { 
					if (graph[i-1][j].data == -1 && graph[i-1][j+1].data == -1 &&
						graph[i][j+1].data == -1 &&
						graph[i+1][j].data == -1 && graph[i+1][j+1].data == -1) {

						graph[i-1][j].data = rand() % 100;
						graph[i-1][j+1].data = rand() % 100;

						graph[i][j+1].data = rand() % 100;

						graph[i+1][j].data = rand() % 100;
						graph[i+1][j+1].data = rand() % 100;
					}

				}
				else if (j == size - 1) { 
					if (graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 && graph[i][j-1].data == -1 &&
						graph[i+1][j-1].data == -1 && graph[i+1][j].data == -1) {

						graph[i-1][j-1].data = rand() % 100;
						graph[i-1][j].data = rand() % 100;

						graph[i][j-1].data = rand() % 100;

						graph[i+1][j-1].data = rand() % 100;
						graph[i+1][j].data = rand() % 100;
					}

				}
				else { 
					if (graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 && graph[i-1][j+1].data == -1 &&
						graph[i][j-1].data == -1 && graph[i][j+1].data == -1 &&
						graph[i+1][j-1].data == -1 && graph[i+1][j].data == -1 && graph[i+1][j+1].data == -1) {

						graph[i-1][j-1].data = rand() % 100;
						graph[i-1][j].data = rand() % 100;
						graph[i-1][j+1].data = rand() % 100;

						graph[i][j-1].data = rand() % 100;
						graph[i][j+1].data = rand() % 100;

						graph[i+1][j-1].data = rand() % 100;
						graph[i+1][j].data = rand() % 100;
						graph[i+1][j+1].data = rand() % 100;
					}
				}
			}
		}
	}

	return graph;
}

void makeGraph(Vertex*** graph, int size) {
	*graph = (Vertex**)malloc(sizeof(Vertex*) * size);
	srand((unsigned int)time(NULL));

	int i = 0;
	for (; i < size; ++i) {
		Vertex* a = (Vertex*)malloc(sizeof(Vertex) * size);
		int j = 0;
		for (; j < size; ++j) {
			int rando = rand() % 150; 
			if (rando < 100) { 
				a[j].data = rando;
			}
			else {
				a[j].data = -1; 
			}
			a[j].visited = 0;
		}

		(*graph)[i] = a;
	}

	*graph = build(*graph, size);
}

void writeGraphTo(Vertex** graph, char* fname, long int size) {
	FILE* f = fopen(fname, "wb");

    int* arr = (int*)malloc(sizeof(int)* size * size);
    
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			arr[j + i*size] = graph[i][j].data;
		}
	}

	fwrite(arr, sizeof(int), size*size, f);
	fclose(f);
	free(arr);
}

Vertex** readGraphFrom(char* fname, long int size) {
	int* arr = (int*)malloc(sizeof(int) * size * size);

	FILE *fp = fopen(fname, "rb");
	if (fp == NULL) {
		printf("Possible file reading error....\n");
		return NULL;
	}
	
	fread(arr, sizeof(int), size*size, fp);

	Vertex** graph = (Vertex**)malloc(sizeof(Vertex*) * size * size);
	
	for (int i = 0; i < size; ++i) {
		graph[i] = (Vertex*)malloc(sizeof(Vertex)*size);
	
		for (int j = 0; j < size; ++j) {
			graph[i][j].data = arr[j + i*size];
			graph[i][j].visited = 0;
		}
	}

	free(arr);

	return graph;
}

__device__ void bfs_gpu_thread(Vertex* graph, int size, Node curr) {

	Node next = curr; 
	int row = *(next.pos);
	int col = *(next.pos + 1);

	if (graph[row*size + col].data != -1) {
		if (row-1 >= 0) { 
			if (graph[(row-1)*size + col].data != -1 && graph[(row-1)*size + col].visited == 0) {
				graph[(row-1)*size + col].visited = 1;
			}
			if (col-1 >= 0) { 
				if (graph[(row-1)*size + (col-1)].data != -1 && graph[(row-1)*size + (col-1)].visited == 0) {
					graph[(row-1)*size + (col-1)].visited = 1;
				}
				if (graph[row*size + (col-1)].data != -1 && graph[row*size + (col-1)].visited == 0) {
					graph[row*size + (col-1)].visited = 1;
				}
			}
			if (col+1 < size) { 
				if (graph[(row-1)*size + (col+1)].data != -1 && graph[(row-1)*size + (col+1)].visited == 0) {
					graph[(row-1)*size + (col+1)].visited = 1;
				}
				if (graph[row*size + (col+1)].data != -1 && graph[row*size + (col+1)].visited == 0) {
					graph[row*size + (col+1)].visited = 1;
				}
			}
		}
		if (row+1 < size) { 
			if (graph[(row+1)*size + col].data != -1 && graph[(row+1)*size + col].visited == 0) {
				graph[(row+1)*size + col].visited = 1;
			}
			if (col-1 >= 0) { 
				if (graph[row*size + (col-1)].data != -1 && graph[row*size + (col-1)].visited == 0) {
					graph[row*size + (col-1)].visited = 1;
				}
				if (graph[(row+1)*size + (col-1)].data != -1 && graph[(row+1)*size + (col-1)].visited == 0) {
					graph[(row+1)*size + (col-1)].visited = 1;
				}
			}
			if (col+1 < size) { 
				if (graph[row*size + (col+1)].data != -1 && graph[row*size + (col+1)].visited == 0) {
					graph[row*size + (col+1)].visited = 1;
				}
				if (graph[(row+1)*size + (col+1)].data != -1 && graph[(row+1)*size + (col+1)].visited == 0) {
					graph[(row+1)*size + (col+1)].visited = 1;
				}
			}
		}
	}
}

__global__	void bfs_gpu_kernel(Vertex* graph, int size) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    
	Node startNode;
	int* temp = (int*)malloc(sizeof(*temp) * 2);
	*temp = graph[idx].xpos;
	*(temp+1) = graph[idx].ypos;
	startNode.data = graph[idx].data;
	startNode.pos = temp;
	startNode.next = NULL;

	for (int i = idx; i < size*size; ++i) {
		bfs_gpu_thread(graph, size, startNode);
	}

	free(temp);
	__syncthreads();

}

int main(int argc, char* argv[]) {
    
    if(argc != 2){
        printf("Usage: ./bfsg <file name>\n");
        exit(1);
    } else{
        int SIZE = 100;
        int THREADS_PER_BLOCK = 100;
    
        srand((unsigned int)time(NULL));
    
        char* fname = argv[1];
        int done = 0;
        
        Vertex** g = readGraphFrom(fname, SIZE);

        if (g == NULL) {
            makeGraph(&g, SIZE);
            done = 1;
        }
        
        Vertex* graph_dev = (Vertex*)malloc(sizeof(*graph_dev) * SIZE * SIZE);
        transfer1(g, &graph_dev, SIZE);
    
        
        Vertex* graph_dev_copy;
        
        cudaSetDevice(0);
        
        cudaMalloc((void**)&graph_dev_copy, sizeof(Vertex) * SIZE * SIZE);
    
        
        cudaMemcpy(graph_dev_copy, graph_dev, sizeof(Vertex) * SIZE * SIZE, cudaMemcpyHostToDevice); 
        
        clock_t start = clock();
        bfs_gpu_kernel<<<(SIZE*SIZE)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(graph_dev_copy, SIZE);
        clock_t stop = clock();
    
        float total = (float)((stop - start) / (float)CLOCKS_PER_SEC);
    
        cudaMemcpy(graph_dev, graph_dev_copy, sizeof(Vertex) * SIZE * SIZE, cudaMemcpyDeviceToHost);
        
        transfer2(&g, graph_dev, SIZE);
    
        cudaFree(&graph_dev_copy);
    
        //display(g, SIZE);
    
        printf("\n");
        printf("BFS in GPU took: %f\n", total);
    
        if (done) { 
            writeGraphTo(g, fname, SIZE); 
        }
    
        freeGraph(g, SIZE);
        free(graph_dev);

        return 0;
    }
}