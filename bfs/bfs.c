#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct Node{
	int data;
	int* pos; 
	struct Node* next;
} Node;

typedef struct Vertex{
	int data;
	int visited;
} Vertex;

void freeLL(Node* head){
	Node* curr = head;
	if(head == NULL){
		printf("Linkedlist is empty!\n");
		return;
	}

	do{
		free(curr->pos);
		free(curr);
		curr = curr->next;
	} while(curr != NULL);
}

void push(Node** head, int pushdata, int* pos){ 
	Node* node = malloc(sizeof(Node));
	node->data = pushdata;
	node->pos = pos;
	if(*head == NULL){ 
		node->next = NULL;
	} else{
		node->next = *head;
	}
	*head = node;
	return;
}

Node pop(Node** head){
	Node* temp = *head; 

	if(*head == NULL){ 
		printf("Nothing to pop!\n");
		Node node;
		node.data = -1;
		node.pos = NULL;
		return node;
	} 

	if((*head)->next == NULL){ 
		Node node1;
		node1.data = (*head)->data;
		node1.pos = (*head)->pos;
		*head = NULL;
		free(temp);
		return node1;
	}
	
	Node* curr = (*head)->next;
	while(1){
		if(curr->next == NULL){
			(*head)->next = NULL;
			Node node2;
			node2.data = curr->data;
			node2.pos = curr->pos;
			free(curr);
			*head = temp;
			return node2;
		}
		*head = curr; 
		curr = curr->next;
	}
}

Vertex** build(Vertex** g, int size){ 
    Vertex** graph;
    graph = g;

    for(int i = 0; i < size; ++i){
        for(int j = 0; j < size; ++j){
            if(i == 0){ 
                if(j == 0){ 
                    if(graph[i+1][j].data == -1 && graph[i][j+1].data == -1 && graph[i+1][j+1].data == -1){
                        graph[i+1][j].data = rand() % 100; 
                        graph[i][j+1].data = rand() % 100; 
                        graph[i+1][j+1].data = rand() % 100; 
                    }
                } else if(j == size - 1){ 
                    if(graph[i+1][j].data == -1 && graph[i][j-1].data == -1 && graph[i+1][j-1].data == -1){
                        graph[i+1][j].data = rand() % 100; 
                        graph[i][j-1].data = rand() % 100; 
                        graph[i+1][j-1].data = rand() % 100; 
                    }
                } else{ 
                    if(graph[i][j-1].data == -1 && graph[i][j+1].data == -1 &&
                        graph[i+1][j-1].data == -1 && graph[i+1][j].data == -1 && graph[i+1][j+1].data == -1){

                        graph[i][j-1].data = rand() % 100; 
                        graph[i][j+1].data = rand() % 100; 
                        graph[i+1][j-1].data = rand() % 100;
                        graph[i+1][j].data = rand() % 100; 
                        graph[i+1][j+1].data = rand() % 100; 

                    }
                }
            } else if(i == size - 1){
                if(j == 0){ 
                    if(graph[i-1][j].data == -1 && graph[i][j+1].data == -1 && graph[i-1][j+1].data == -1){
                        graph[i-1][j].data = rand() % 100; 
                        graph[i][j+1].data = rand() % 100; 
                        graph[i-1][j+1].data = rand() % 100; 
                    }
                } else if(j == size - 1){ 
                    if(graph[i-1][j].data == -1 && graph[i][j-1].data == -1 && graph[i-1][j-1].data == -1){
                        graph[i-1][j].data = rand() % 100; 
                        graph[i][j-1].data = rand() % 100; 
                        graph[i-1][j-1].data = rand() % 100; 
                    }
                } else{ 
                    if(graph[i][j-1].data == -1 && graph[i][j+1].data == -1 &&
                        graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 && 
                        graph[i-1][j+1].data == -1){

                        graph[i][j-1].data = rand() % 100; 
                        graph[i][j+1].data = rand() % 100; 
                        graph[i-1][j-1].data = rand() % 100;
                        graph[i-1][j].data = rand() % 100; 
                        graph[i-1][j+1].data = rand() % 100; 

                    }
                }
            } else{
                if(j == 0){ 
                    if(graph[i-1][j].data == -1 && graph[i-1][j+1].data == -1 && 
                        graph[i][j+1].data == -1 && graph[i+1][j].data == -1 && 
                        graph[i+1][j+1].data == -1){

                        graph[i-1][j].data = rand() % 100;
                        graph[i-1][j+1].data = rand() % 100;

                        graph[i][j+1].data = rand() % 100;

                        graph[i+1][j].data = rand() % 100;
                        graph[i+1][j+1].data = rand() % 100;
                    }

                } else if(j == size - 1){ 
                    if(graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 &&
                        graph[i][j-1].data == -1 && graph[i+1][j-1].data == -1 && 
                        graph[i+1][j].data == -1){

                        graph[i-1][j-1].data = rand() % 100;
                        graph[i-1][j].data = rand() % 100;

                        graph[i][j-1].data = rand() % 100;

                        graph[i+1][j-1].data = rand() % 100;
                        graph[i+1][j].data = rand() % 100;
                    }

                } else{ 
                    if(graph[i-1][j-1].data == -1 && graph[i-1][j].data == -1 && 
                        graph[i-1][j+1].data == -1 && graph[i][j-1].data == -1 && 
                        graph[i][j+1].data == -1 && graph[i+1][j-1].data == -1 && 
                        graph[i+1][j].data == -1 && graph[i+1][j+1].data == -1){

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

void printg(Vertex** graph, int size){
    for(int i = 0;i < size; ++i){
        for(int j = 0; j < size; ++j){
            if(graph[i][j].data != -1){
                if(graph[i][j].data < 10){
                    printf("%d  ", graph[i][j].data);
                } 
                
                else{
                    printf("%d ", graph[i][j].data);
                }
            } 
            
            else{
                printf("   ");
            }
        }
        printf("\n");
    }

    printf("\n");
}

void makeg(Vertex*** graph, int size){
    *graph = malloc(sizeof(Vertex*) * size);
    srand(time(NULL));

    for(int i = 0; i < size; ++i){
        Vertex* v = malloc(sizeof(Vertex) * size);
        for(int j = 0; j < size; ++j){
            int rnd = rand() % 150; 
            (rnd < 100) ? (v[j].data = rnd) : (v[j].data = -1);
            v[j].visited = 0;
        }

        (*graph)[i] = v;
    }

    *graph = build(*graph, size);
}

void freeGraph(Vertex** graph, int size){
    for(int i = 0; i < size; ++i){
        free(graph[i]);
    }

    free(graph);
}

void writeGraphTo(Vertex** graph, char* fname, long int size){
    FILE* fp = fopen(fname, "wb");

    int* arr = malloc(sizeof(int)* size * size);

    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            arr[j + i*size] = graph[i][j].data;
        }
    }

    fwrite(arr, sizeof(int), size*size, fp);
    fclose(fp);
    free(arr);
}

Vertex** readGraphFrom(char* fname, long int size){
    int* arr = malloc(sizeof(int) * size * size);

    FILE *fp = fopen(fname, "rb");
    if(fp == NULL){
        printf("Something went wrong here.... Possible error with file.\n");
        return NULL;
    }

    fread(arr, sizeof(int), size*size, fp);

    Vertex** graph = malloc(sizeof(Vertex*) * size * size);
    
    for (int i = 0; i < size; ++i){
        graph[i] = malloc(sizeof(Vertex)*size);
        for (int j = 0; j < size; ++j){
            graph[i][j].data = arr[j + i*size];
            graph[i][j].visited = 0;
        }
    }

    fclose(fp);
    free(arr);

    return graph;
}

Node getStartingPosition(Vertex*** graph, int size){
	for(int i = 0; i < size; ++i){
		for(int j = 0; j < size; ++j){
			if((*graph)[i][j].data != -1){
				Node startNode;
				int* temp = malloc(sizeof(*temp)*2);
				*temp = i;
				*(temp+1) = j;
				startNode.data = (*graph)[i][j].data;
				(*graph)[i][j].visited = 1;
				startNode.pos = temp;
				startNode.next = NULL;
				return startNode;
			}
		}
	}
}

void pushn(Vertex*** graph, Node** head, int row, int col, int** indexer){
	int data = (*graph)[row][col].data;
	**indexer = (*graph)[row][col].data;
	*indexer = *indexer + 1;
	int* temp = malloc(sizeof(*temp) * 2);
	*temp = row;
	*(temp+1) = col;
	push(head, data, temp);
}


static void bfsr(Vertex*** graph, int size, Node startNode, Node** head, int** indexer){
	int seen = 1; 
	Node next;

	int row = *(startNode.pos);
	int col = *(startNode.pos+1);

	if(row - 1 >= 0){ 
		if((*graph)[row-1][col].data != -1 && (*graph)[row-1][col].visited == 0){
			(*graph)[row-1][col].visited = 1;
			pushn(graph, head, row-1, col, indexer);
			seen = 0;
		}
		if(col - 1 >= 0){ 
			if((*graph)[row-1][col-1].data != -1 && (*graph)[row-1][col-1].visited == 0){
				(*graph)[row-1][col-1].visited = 1;
				pushn(graph, head, row-1, col-1, indexer);
				seen = 0;
			}
			if((*graph)[row][col-1].data != -1 && (*graph)[row][col-1].visited == 0){
				(*graph)[row][col-1].visited = 1;
				pushn(graph, head, row, col-1, indexer);
				seen = 0;
			}
		} 
		if(col + 1 < size){ 
			if((*graph)[row-1][col+1].data != -1 && (*graph)[row-1][col+1].visited == 0){
				(*graph)[row-1][col+1].visited = 1;
				pushn(graph, head, row-1, col+1, indexer);
				seen = 0;
			}
			if((*graph)[row][col+1].data != -1 && (*graph)[row][col+1].visited == 0){
				(*graph)[row][col+1].visited = 1;
				pushn(graph, head, row, col+1, indexer);
				seen = 0;
			}
		}
	}
	if(row + 1 < size){ 
		if((*graph)[row+1][col].data != -1 && (*graph)[row+1][col].visited == 0){
			(*graph)[row+1][col].visited = 1;
			pushn(graph, head, row + 1, col, indexer);
			seen = 0;
		}
		if(col - 1 >= 0){ 
			if((*graph)[row][col-1].data != -1 && (*graph)[row][col-1].visited == 0){
				(*graph)[row][col-1].visited = 1;
				pushn(graph, head, row, col-1, indexer);
				seen = 0;
			}
			if((*graph)[row+1][col-1].data != -1 && (*graph)[row+1][col-1].visited == 0){
				(*graph)[row+1][col-1].visited = 1;
				pushn(graph, head, row + 1, col-1, indexer);
				seen = 0;
			}
		}
		if(col + 1 < size){ 
			if((*graph)[row][col+1].data != -1 && (*graph)[row][col+1].visited == 0){
				(*graph)[row][col+1].visited = 1;
				pushn(graph, head, row, col+1, indexer);
				seen = 0;
			}
			if((*graph)[row+1][col+1].data != -1 && (*graph)[row+1][col+1].visited == 0){
				(*graph)[row+1][col+1].visited = 1;
				pushn(graph, head, row + 1, col+1, indexer);
				seen = 0;
			}
		}
	}

	if(!seen){ 
		next = pop(head);
		bfsr(graph, size, next, head, indexer);
	} else{
		if(*head != NULL){
			next = pop(head);
			bfsr(graph, size, next, head, indexer);
		} else{
			return;
		}
	}
}

void bfsi(Vertex*** graph, int size, Node curr, Node** head, int** indexer) {
        Node next = curr; 
    
        while (1) {
            int seen = 1; 
    
            int row =*(next.pos);
            int col =*(next.pos+1);
    
            if(row -1 >= 0) { 
                if((*graph)[row-1][col].data != -1 && (*graph)[row-1][col].visited == 0) {
                    (*graph)[row-1][col].visited = 1;
                    pushn(graph, head, row-1, col, indexer);
                    seen = 0;
                }
                if(col-1 >= 0) { 
                    if((*graph)[row-1][col-1].data != -1 && (*graph)[row-1][col-1].visited == 0) {
                        (*graph)[row-1][col-1].visited = 1;
                        pushn(graph, head, row-1, col-1, indexer);
                        seen = 0;
                    }
                    if((*graph)[row][col-1].data != -1 && (*graph)[row][col-1].visited == 0) {
                        (*graph)[row][col-1].visited = 1;
                        pushn(graph, head, row, col-1, indexer);
                        seen = 0;
                    }
                } 
                if (col + 1 < size) { 
                    if((*graph)[row-1][col+1].data != -1 && (*graph)[row-1][col+1].visited == 0) {
                        (*graph)[row-1][col+1].visited = 1;
                        pushn(graph, head, row-1, col+1, indexer);
                        seen = 0;
                    }
                    if((*graph)[row][col+1].data != -1 && (*graph)[row][col+1].visited == 0) {
                        (*graph)[row][col+1].visited = 1;
                        pushn(graph, head, row, col+1, indexer);
                        seen = 0;
                    }
                }
            }
            if(row+ 1 < size) { 
                if((*graph)[row+1][col].data != -1 && (*graph)[row+1][col].visited == 0) {
                    (*graph)[row+1][col].visited = 1;
                    pushn(graph, head, row+1, col, indexer);
                    seen = 0;
                }
                if(col-1 >= 0) { 
                    if((*graph)[row][col-1].data != -1 && (*graph)[row][col-1].visited == 0) {
                        (*graph)[row][col-1].visited = 1;
                        pushn(graph, head, row, col-1, indexer);
                        seen = 0;
                    }
                    if((*graph)[row+1][col-1].data != -1 && (*graph)[row+1][col-1].visited == 0) {
                        (*graph)[row+1][col-1].visited = 1;
                        pushn(graph, head, row+1, col-1, indexer);
                        seen = 0;
                    }
                }
                if(col + 1 < size) { 
                    if((*graph)[row][col+1].data != -1 && (*graph)[row][col+1].visited == 0) {
                        (*graph)[row][col+1].visited = 1;
                        pushn(graph, head, row, col+1, indexer);
                        seen = 0;
                    }
                    if ((*graph)[row+1][col+1].data != -1 && (*graph)[row+1][col+1].visited == 0) {
                        (*graph)[row+1][col+1].visited = 1;
                        pushn(graph, head, row+1, col+1, indexer);
                        seen = 0;
                    }
                }
            }
    
   
            if(!seen) { 
                next = pop(head);
            } else {
                if (*head != NULL) {
                    next = pop(head);
                } else {
                    break;
                }
            }
        }
    }

int main(int argc, char** argv){
    if(argc != 3){
        printf("Usage: ./bfs <file name> <size>\n");
        exit(1);
    } else{
        char* fname = argv[1];
        int size = atoi(argv[2]);

        Node* head = NULL;
        clock_t start, stop;
        
        int done = 0;
        
        Vertex** x =  readGraphFrom(fname, size);
        if(x == NULL){
             makeg(&x, size);
             done = 1;
        }
    
        Node startNode = getStartingPosition(&x, size);
        
        int* bbuf = malloc(sizeof(*bbuf) * (size*size));
         
        for (int i = 0; i < size*size; ++i){ 
             bbuf[i] = -1;
        }
         
         int* cbuff = bbuf;
    
         start = clock();
         bfsr(&x, size, startNode, &head, &cbuff);
         stop = clock();
         float total = (float)((stop-start)/(float)CLOCKS_PER_SEC);
    
        printf("\n");
    
        for(int i; i < size*size; ++i){
            if(bbuf[i] == -1){
                break;
            }
            if(i % 10 != 9){
                printf("%d, ", bbuf[i]);
            } 
            else{
                printf("%d\n", bbuf[i]);
            }
        }

        printf("\n");
    
        printg(x, size);
     
        if(done) writeGraphTo(x, fname, size); 
         
    
        freeGraph(x, size);
        freeLL(head);
         
        printf("\n");
    
        printf("time taken: %f\n", total);
    
        return 0;
    }
}