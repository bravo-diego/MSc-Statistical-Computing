# include <stdio.h>
# include <stdlib.h>
# include <omp.h> // OpenMP library

int main(void){

  omp_set_num_threads(2); // set the number of threads
  #pragma omp parallel  
  {
    #pragma omp for // the index of the uppermost for loop is divides among the threads; i.e. when index of i is same the thread id is also the same
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        printf("i = %d, j = %d, thread id = %d \n", i, j, omp_get_thread_num());
      }
    }
  
    #pragma omp for collapse(2) // for the same index, there can be different threads ids; i.e. the combinations of i and j are divided among the threads
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        printf("i = %d, j = %d, thread id = %d \n", i, j, omp_get_thread_num());
      }
    }
  }

  return(0);
}
