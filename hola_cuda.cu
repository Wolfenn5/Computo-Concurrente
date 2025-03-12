#include <stdio.h>
#include <cuda_runtime.h>



int main(int argc, char const *argv[])
{
    cudaDeviceProp propiedades; //variable de tipo cudaDeviceProp que pertenece a cuda_runtime
    
    int idDispositivo;
    // se quiere obtener el id del dispositivo GPU
    cudaGetDevice(&idDispositivo); // se obtiene la informacion del dispositivo (GPU)
    cudaGetDeviceProperties(&propiedades, idDispositivo); // sacar todas las propiedades del dispositivo

    printf("\nInformacion de la GPU %d: %s",idDispositivo, propiedades.name); // name es un miebro de cudaDeviceProp que sirve para obtener el nombre de la GPU

    // Impresion de informacion del dispositivo
    printf("\nNumero maximo de procesadores(SM): %d", propiedades.multiProcessorCount);
    printf("\nNumero maximo de hilos por bloque: %d", propiedades.maxThreadsPerBlock);
    printf("\nNumero maximo de hilos por SM: %d", propiedades.maxThreadsPerMultiProcessor);
    printf("\nNumero maximo de bloques por grid: %d", propiedades.maxGridSize[0]);
    printf("\nNumero maximo de hilos por dimension del bloque: (%d %d %d)", propiedades.maxThreadsDim[0], propiedades.maxThreadsDim[1], propiedades.maxThreadsDim[2]);
    printf("\nTamaño de la memoria global: %.ld GB", propiedades.totalGlobalMem);
    printf("\nTamaño de memoria compartida (por bloques) %d", propiedades.sharedMemPerBlock);
    return 0;
}
