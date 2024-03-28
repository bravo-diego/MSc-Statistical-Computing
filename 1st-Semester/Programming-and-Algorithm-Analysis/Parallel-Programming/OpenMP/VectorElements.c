# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <omp.h> // OpenMP library

int main(void){
  
  srand(time(NULL)); // use current time as seed for random generator
  
  int sum = 0;
  int N = 1000000;
  int len = 0;
  
  int (*V)= malloc(N*sizeof(int)); // allocate memory using malloc()
  
  for(int i = 0; i < N; i++){
    V[i] = rand() % N;
    len += 1;
  }
  
  printf("Length of vector V: %d\n", len); // verify if vector V has the elements specified in variable N
  
  clock_t begin = clock(); // start time
  
  #pragma omp parallel 
  {
    #pragma omp for reduction(+:sum) // divides loop iterations between the spawned threads; reduction specifies how to combine local copies of a variable in different threads into a single copy at the master when threads exit
    for(int j = 0; j < N; j++){
      sum += V[j];
    }
  }
  
  clock_t end = clock(); // end time
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // execution time
  
  printf("Sum of elements in vector V: %d\n", sum); 
  printf("Elapsed: %lf seconds.\n", time_spent); 
  
  free(V); // free allocated memory
  
  return(0);
}

