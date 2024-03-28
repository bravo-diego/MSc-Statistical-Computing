# include <stdio.h>
# include <omp.h> // OpenMP library

int main(void){

  int nthreads, tid;
  omp_set_num_threads(4); // set the number of threads
	
  #pragma omp parallel private(tid) // a private variable has a different address in the execution context of every thread
    {
      tid = omp_get_thread_num(); // returns the number of the currently executing thread within the team
      printf("Hello world, thread number %d\n", tid);
	
      if (tid == 0){
        nthreads = omp_get_num_threads(); // returns the number of threads in the team currently executing the parallel region from which it is called
        printf("Number of threads = %d\n", nthreads);
      }
    } // each thread executes a copy of the code within the structured block
  return(0);
}
