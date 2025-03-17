#include <stdio.h>

/* de momento este programa es paralelo. CUDA por naturaleza es asincrono */

// se va a trabajar con 9 hilos

__global__ void kernel ()
{
    printf("\nHola desde el bloque %d, hilo %d\n", blockIdx.x, threadIdx.x); // blockidx y threadidx son para obtener el bloque y el hilo de ese bloque en eje x (los 2 hilo y bloque)
    __syncthreads(); // sincronizar los hilos dentro del mismo bloque
    printf("\nHola despues de la sincronizacion y soy el bloque %d, hilo%d \n", blockIdx.x, threadIdx.x); 
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


    cudaDeviceSynchronize(); // es el equivalente a .join en hilos. Es importante utilizarlo para no trabar la GPU

    printf("\nEn total puede ejecutar %d hilos\n", num_bloques*num_hilos); // numero de hilos totales
    return 0;
}
