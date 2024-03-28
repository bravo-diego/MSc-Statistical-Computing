# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <omp.h> // OpenMP library

int main(void){
  
  srand(time(NULL)); // use current time as seed for random generator
  
  int N = 1000; 
  int M = 2000;
  double a = 0.1;
  int rows = 0;
  int columns = 0;
  int tid;
  
  int (*A)[M] = malloc(M*sizeof(*A)); // allocate memory using malloc()
  int (*B)[M] = malloc(M*sizeof(*B)); 
  int (*C1)[M] = malloc(M*sizeof(*C1)); // C := C[i][j] = A[i][j] + B[N - i - 1][M - j - 1]
  double (*C2)[M] = malloc(M*sizeof(*C2)); // C := C[i][j] = (a)A[i][j] + (1 - a)B[N - i - 1][M - j - 1]; with a in [0, 1]
  
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      A[i][j] = (rand() % N*M) + 1;
      columns += 1; // verify dimensions of matrix A
    }
    rows += 1; // verify dimensions of matrix A
  }

  printf("dim matrix A %dx%d\n", rows, columns/N); 

  rows = 0; // restart no. rows
  columns = 0; // restart no. columns

  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      B[i][j] = (rand() % N*M) + 1;
      columns += 1; // verify dimensions of matrix B
    }
    rows += 1; // verify dimensions of matrix B
  }
  
  printf("dim matrix B %dx%d\n", rows, columns/N); 

  rows = 0; // restart no. rows
  columns = 0; // restart no. columns
  
  clock_t begin = clock(); // start time
  
  omp_set_num_threads(4); // set the number of threads
  #pragma omp parallel default(none) shared(N, M, A, B, C1, C2, a) private(tid)
    {
      tid = omp_get_thread_num(); // returns the number of the currently executing thread within the team
      #pragma omp for collapse(2) // for the same index, there can be different threads ids; i.e. the combinations of i and j are divided among the threads
      for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
          C1[i][j] = A[i][j]+B[N - i- 1][M - j - 1]; // matrix C_{1}
          printf("Thread id: %d, i: %d, j: %d\n", tid, i, j);
        }
      }
      
      #pragma omp for collapse(2)
      for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
        C2[i][j] = (a*A[i][j])+((1- a)*B[N - i- 1][M - j - 1]); // matrix C_{2}
        printf("Thread id: %d, i: %d, j: %d\n", tid, i, j);
        }
      }
      
    }
    
  clock_t end = clock(); // end time
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // execution time
  
  /* printf("Matrix C_{1}.\n");
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      printf("%d ", C1[i][j]); // print elements matrix C_{1}
    }
    printf("\n");
  }
  
  printf("Matrix C_{2}.\n");
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      printf("%.2lf ", C2[i][j]); // print elements matrix C_{2}
    }
    printf("\n");
  } */
  
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      columns += 1; // verify dimensions of matrices C_{1} and C_{2}
    }
    rows += 1; // verify dimensions of matrices C_{1} and C_{2}
  }
  
  printf("dim Matrices C_{i} %dx%d\n", rows, columns/N);
  printf("Elapsed: %lf seconds.\n", time_spent); 

  free(A); // free allocated memory
  free(B);
  free(C1);
  free(C2);

  return(0);
}

