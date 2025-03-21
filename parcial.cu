#include <stdio.h>
#include <cuda_runtime.h>


/* Ejemplo sencillo de dim3 con gridsize y blocksize*/

// __global__ void kernel()
// {
// int threadId = blockIdx.x * blockDim.x + threadIdx.x;
// printf("Hilo ID: %d\n", threadId);
// }



// int main()
// {
// dim3 gridSize(2, 1, 1); //malla de 2 bloques en x, 1 bloque en y, 1 bloque en z
// // el numero total de bloques en la cuadricula es 2x1x1= 2

// dim3 blockSize(3, 1, 1); //bloques de 3 hilos en x, 1 hilo en y, 1 hilo en z
// // cada bloque tendra 3x1x1= 3 hilos


// // Asi el numero total de hilos en toda la cuadricula va a ser de 2x3= 6
// kernel<<<gridSize, blockSize>>>();
// cudaDeviceSynchronize();
// return 0;
// }



int main(int argc, char const *argv[])
{
    int N=atoi(argv[1]); // parametro del main para tamaño


    // Sacar propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);



    // calcular tamaño optimo del bloque
    int tam_bloque= propiedades.maxThreadsPerBlock; 

    // Calcular numero de bloques necesario
    int num_bloques= (N+tam_bloque-1) / tam_bloque;


    // Definir el tamaño de la malla y el bloque
    dim3 tamanio_bloque(num_bloques,num_bloques);
    dim3 tamanio_malla((N+tam_bloque-1) / tam_bloque,(N+tam_bloque-1) / tam_bloque);


    // Lanzamiento del kernel

    
    return 0;
}
