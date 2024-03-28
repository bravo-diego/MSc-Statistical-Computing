# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <omp.h> // OpenMP library

int main(void){

  srand(time(NULL)); // use current time as seed for random generator

  int i, j, k;
  int N = 2000; // matrix dimension
  int rows = 0; // no. rows
  int columns = 0; // no. columns
  int tid; // thread id
  
  int (*A)[N] = malloc(N*sizeof(*A)); // allocate memory using malloc()
  int (*B)[N] = malloc(N*sizeof(*B)); 
  int (*C)[N] = malloc(N*sizeof(*C)); // matrix C := AB
  
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      A[i][j] = rand() % N;
      columns += 1; // verify dimensions of matrices A and B
    }
    rows += 1; // verify dimensions of matrices A and B
  }

  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      B[i][j] = rand() % N;
    }
  }
  
  printf("dim matrix A %dx%d; dim matrix B %dx%d\n", rows, columns/N, rows, columns/N); 
  
  rows = 0; // restart no. rows
  columns = 0; // restart no. columns
  
  clock_t begin = clock(); // start time
  
  #pragma omp parallel shared(A, B, C, N) private(i, j, k, tid) 
  { 
    tid = omp_get_thread_num(); // returns the number of the currently executing thread within the team
    #pragma omp for reduction(+:rows, columns)
    for(int i = 0; i < N; i ++){
      for(int j = 0; j < N; j++){
        C[i][j] = 0;
        for(int k = 0; k < N; k ++){
          C[i][j] += A[i][k]*B[k][j];
          printf("Thread id: %d, i: %d, j: %d, k: %d\n", tid, i, j, k);
        }
        columns += 1; // verify dimensions of matrix C
      }
      rows += 1; // verify dimensions of matrix C
    }
  }
  
  clock_t end = clock(); // end time
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // execution time
  
  /* for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      printf("%d ", C[i][j]); // print elements matrix C 
    }
    printf("\n");
  } */
  
  printf("dim Matrix C %dx%d\n", rows, columns/N);
  printf("Elapsed: %lf seconds.\n", time_spent); 
  
  free(A); // free allocated memory
  free(B);
  free(C);

  return(0);
}
