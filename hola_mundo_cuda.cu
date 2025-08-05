#include <stdio.h>

/* 
Para compilar se utiliza nvcc de la siguiente forma:

nvcc hola_mundo_cuda.cu -o hola_cuda

./hola_cuda
*/


/* de momento este programa es paralelo */

// se va a trabajar con 9 hilos

__global__ void kernel ()
{
    printf("\nHola desde el bloque %d, hilo %d\n", blockIdx.x, threadIdx.x); // blockidx y threadidx son para obtener el bloque y el hilo de ese bloque en eje x (los 2 hilo y bloque)
    __syncthreads(); // sincronizar los hilos dentro del mismo bloque
    printf("\nHola despues de la sincronizacion y soy el bloque %d, hilo%d \n", blockIdx.x, threadIdx.x); 
}



int main(int argc, char const *argv[])
{
    int num_bloques= 3;
    int num_hilos= 3;
    printf("\nEjecutando kernel en CUDA \n");

    kernel<<<num_bloques,num_hilos>>>();
    cudaDeviceSynchronize(); 
    return 0;
}
