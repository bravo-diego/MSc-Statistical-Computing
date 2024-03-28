# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <omp.h> // OpenMP library

int main(void){
  
  int N = 1000000; // vector dimension
  int Vlen = 0;
  int S1len = 0;
  int S2len = 0;
  int tid;
  
  int (*V)= malloc(N*sizeof(int)); // allocate memory using malloc()
  int (*S1)= malloc((N - 1)*sizeof(int)); // S := S[i] = V[i] + V[i+1]
  int (*S2)= malloc((N - 2)*sizeof(int)); // S := S[i] = V[i+1] + V[i-1]/2 
  
  for (int i = 0; i < N; i++){
    V[i] = i + 1;
    Vlen += 1;
  }
  
  printf("Length of vector V: %d\n", Vlen); // verify if vector V has the number of elements specified in variable N
  
  clock_t begin = clock(); // start time
  
  #pragma omp parallel default(none) shared(V, N, S1, S2) private(tid) reduction(+:S1len, S2len) // reduction specifies how to combine local copies of a variable in different threads into a single copy at the master when threads exit
    {
      tid = omp_get_thread_num(); // returns the number of the currently executing thread within the team
      #pragma omp for // divides loop iterations between the spawned threads
      for (int i = 0; i < N - 1; i++){
        S1[i] = V[i] + V[i + 1];
        S1len += 1; // verify length vector S_{1} N - 1 elements
        printf("Thread id: %d, i: %d\n", tid, i);
      }
      
      #pragma omp for // divides loop iterations between the spawned threads
      for (int i = 1; i < N - 1; i++){
        S2[i - 1] = ((V[i - 1] + V[i + 1])/2);
        S2len += 1; // verify length vector S_{2} N - 2 elements
        printf("Thread id: %d, i: %d\n", tid, i);
      }
    }
    
  clock_t end = clock(); // end time
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // execution time
  
  /* printf("Length of vector S_{1}: %d\n", S1len); // verify if vector S_{1} has N - 1 elements
  for(int i = 0; i < N - 1; i++){
    printf("%d \n", S1[i]); // print vector S_{1}
  }
  
  printf("Length of vector S_{2}: %d\n", S2len); // verify if vector S_{2} has N - 2 elements
  for(int i = 0; i < N - 2; i++){
    printf("%d \n", S2[i]); // print vector S_{2}
  } */
  
  printf("Length of vector S_{1}: %d\n", S1len); // verify length vector S_{1} N - 1 elements
  printf("Length of vector S_{2}: %d\n", S2len); // verify length vector S_{2} N - 2 elements
  printf("Elapsed: %lf seconds.\n", time_spent); 
  
  free(V); // free allocated memory
  free(S1);
  free(S2);

  return(0);
}

