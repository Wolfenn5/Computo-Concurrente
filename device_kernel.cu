#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


/* Este programa usa funciones global llamadas desde el host hacia el dispositivo y funciones device llamadas desde el dispositivo hacia el dispositivo*/


__device__ int suma (int a, int b)
{
    return a+b;
}


__device__ int multiplicacion (int a, int b)
{
    return a*b;
}




__global__ void operaciones (int *arreglo, int tamanio)
{
    int id= threadIdx.x; // como hay un unico bloque y un unico idBloque queda   0*0+threadIdx
    int a= id+1;
    int b= id+2;


    arreglo[id]= suma(a,b);
    arreglo[id+tamanio]= multiplicacion(a,b); // [a+b, a+b, a*b, a*b]
}


int main(int argc, char const *argv[])
{
    // Parametro del main
    int tamanio= atoi(argv[1]); // tamaño del arreglo


    int *arreglo_dispositivo;


    cudaMalloc(&arreglo_dispositivo, 2*tamanio*sizeof(int));


    operaciones<<<1, tamanio>>>(arreglo_dispositivo, tamanio);
    cudaDeviceSynchronize();

    int * arreglo_host= (int *)malloc(tamanio*2*sizeof(int));
    cudaMemcpy(arreglo_host, arreglo_dispositivo, tamanio*2*sizeof(int), cudaMemcpyDeviceToHost);


    for (int i=0; i<tamanio; i++)
    {
        printf("\nLa suma de %d y %d es: %d", i+1, i+2, arreglo_host[i]);
        printf("\nLa multiplicacion de %d y %d es: %d", i+1, i+2, arreglo_host[i+tamanio]);
    }
    printf("\n");

    return 0;
}
