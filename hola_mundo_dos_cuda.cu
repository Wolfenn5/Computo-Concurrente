#include <stdio.h>

/* de momento este programa es paralelo */

// se va a trabajar con 9 hilos

__global__ void kernel ()
{
    printf("\nHola desde el bloque %d, hilo %d\n", blockIdx.x, threadIdx.x); // blockidx y threadidx son para obtener el bloque y el hilo de ese bloque en eje x (los 2 hilo y bloque)
}



int main(int argc, char const *argv[])
{
    int num_bloques;
    int num_hilos;
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);

    num_bloques= propiedades.multiProcessorCount; // numero de multiprocesadores que soporta la GPU (como los bloques)

    num_hilos= propiedades.maxThreadsPerBlock; // numero maximo de hilos por bloque
    printf("\nEjecutando kernel en CUDA \n");

    kernel<<<num_bloques,num_hilos>>>();
    cudaDeviceSynchronize(); 
    return 0;
}
