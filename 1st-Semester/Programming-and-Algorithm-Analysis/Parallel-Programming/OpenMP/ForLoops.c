# include <stdio.h>
# include <omp.h> // OpenMP library

int main(void){
  
  int nthreads, tid;
  
  omp_set_num_threads(3); // set the number of threads
  #pragma omp parallel private(tid)
    {
      tid = omp_get_thread_num(); // returns the number of the currently executing thread within the team
      printf("Hello world from %d\n", tid);
      #pragma omp for // divides loop iterations between the spawned threads
      for(int i = 0; i <= 4; i++){
        printf("iteration %d by %d\n", i, tid);
      }
    } // all threads join master thread and terminates
  return(0);
}

