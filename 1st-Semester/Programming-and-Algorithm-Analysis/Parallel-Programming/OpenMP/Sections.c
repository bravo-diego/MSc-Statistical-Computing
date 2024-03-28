#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // OpenMP library

int main(void){
  
  srand(time(NULL)); // use current time as seed for random generator
  
  int N = 700; // matrix dimensions
  int M = 800;
  int K = 900;

  // Allocate memory using malloc()
  int **A = (int **)malloc(M * sizeof(int *)); 
  int **B = (int **)malloc(K * sizeof(int *));
  int **D = (int **)malloc(N * sizeof(int *));
  int **C = (int **)malloc(M * sizeof(int *));
  int **E = (int **)malloc(K * sizeof(int *));

  for (int i = 0; i < M; i++){
    A[i] = (int *)malloc(K * sizeof(int));
  }

  for (int i = 0; i < K; i++){
    B[i] = (int *)malloc(N * sizeof(int));
  }

  for (int i = 0; i < N; i++){
    D[i] = (int *)malloc(M * sizeof(int));
  }
   
  for (int i = 0; i < M; i++){
    C[i] = (int *)malloc(N * sizeof(int));
  }
  
  for (int i = 0; i < K; i++){
    E[i] = (int *)malloc(M * sizeof(int));
  }
  
  for(int i = 0; i < M; i++){
    for(int j = 0; j < K; j++){
      A[i][j] = (rand() % 10) + 1;
    } 
  }
  
  for(int i = 0; i < K; i++){
    for(int j = 0; j < N; j++){
      B[i][j] = (rand() % 10) + 1;
    } 
  }
  
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      D[i][j] = (rand() % 10) + 1;
    } 
  }
  
  clock_t begin = clock(); // start time
  #pragma omp parallel shared (A, B, C, D, E, M, N, K) 
  {
    #pragma omp sections nowait // sections directive enables specification of task parallelism (i.e. sections construct gives a different structured block to each thread); each section is executed by a single thread
    {
      #pragma omp section
      {
        printf("tid = %d \n", omp_get_thread_num());
        for(int i = 0; i < M; i ++){ // number of rows in matrix A
          for(int j = 0; j < N; j++){ // number of columns in matrix B
            C[i][j] = 0;
            for(int k = 0; k < K; k ++){ // entry C[i][j] is obtained by multiplying term-by-term the entries of the ith row of A and the jth column of B and summing these K products
              C[i][j] += A[i][k]*B[k][j]; // matrix C has to be an M by N matrix
            }
          }
        }
      }
      
      #pragma omp section
      {
        printf("tid = %d \n", omp_get_thread_num());
        for(int i = 0; i < K; i ++){ // number of rows in matrix B
          for(int j = 0; j < M; j++){ // number of columns in matrix D
            E[i][j] = 0;
            for(int k = 0; k < N; k ++){ // entry E[i][j] is obtained by multiplying term-by-term the entries of the ith row of B and the jth column of D and summing these N products
              E[i][j] += B[i][k]*D[k][j]; // matrix E has to be an K by M matrix
            }
          }
        }
      }
      
    }
  }
  
  clock_t end = clock(); // end time
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // execution time
  
  /*for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      printf("%d ", C[i][j]); // print elements matrix C
    }
    printf("\n");
  }
  
  for(int i = 0; i < K; i++){
    for(int j = 0; j < M; j++){
      printf("%d ", E[i][j]); // print elements matrix E
    }
    printf("\n");
  }*/
  
  printf("Elapsed: %lf seconds.\n", time_spent); 
  
  // Free allocated memory
  for (int i = 0; i < M; i++){
    free(A[i]);
    free(C[i]);
  }

  for (int i = 0; i < K; i++){
    free(B[i]);
    free(E[i]);
  }

  for (int i = 0; i < N; i++){
    free(D[i]);
  }

  free(A);
  free(B);
  free(D);
  free(C);
  free(E);

    return 0;
}
