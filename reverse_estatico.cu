#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>



__device__ void imprimeVectorDevice (int*d)
{
    printf("%d ",d[blockIdx.x * blockDim.x + threadIdx.x]); 
}


__global__ void reverse_estatico (int *d, int n)
{
    // La directiva __shared__ sirve para especificar que la variable va a ser almacenada en el SM (shared memmory)
    __shared__ int s[1000];
    int t= blockIdx.x * blockDim.x + threadIdx.x;
    if (t<n)
    {
        int tr= n - t - 1; //la posision contraria a t  Si n es 1000 cuando t es 0   tr= 1000-0-1=999  por los arreglos que empieza desde 0 hasta n-1    si t=999  tr= 1000-999-1=0
        // Copiar desde el device (RAM de la GPU) hasta el SM
        s[t]= d[t];

        __syncthreads(); // esperar a que cada uno de los hilos termine de acceder a memoria compartida
        d[t]= s[tr]; // copiar cruzado
        imprimeVectorDevice(d);
    }
}




int main(int argc, char const *argv[])
{
    // Parametro del main para tamaño
    int n= atoi(argv[1]);

    int *arreglo_host=(int*)malloc(sizeof(int)*n);  // Esta variable se aloja en la RAM del host

    int*arreglo_dispositivo; // Esta variable se aloja en la RAM del dispositivo

    for (int i=0; i<n; i++)
    {
        arreglo_host[i]= i+1;
        printf("%d ", arreglo_host[i]);
    }
    printf("\n");
    
    cudaMalloc(&arreglo_dispositivo,sizeof(int)*n);
    cudaMemcpy(arreglo_dispositivo, arreglo_host,sizeof(int)*n,cudaMemcpyHostToDevice);

    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);


    int numeroHilos= propiedades.maxThreadsPerBlock;
    int numeroBloques= (n+numeroHilos-1) / numeroHilos;


    reverse_estatico<<<numeroBloques, numeroHilos>>>(arreglo_dispositivo,n);
    cudaDeviceSynchronize();


    cudaMemcpy(arreglo_host,arreglo_dispositivo,sizeof(int)*n, cudaMemcpyDeviceToHost);


    cudaFree(arreglo_dispositivo);
    return 0;
}
