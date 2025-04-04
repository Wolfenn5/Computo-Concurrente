#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/* La directiva __host__ sirve para ejecutar funciones en la cpu, en si no tiene mucho sentido su uso ya que seria equivalente a decid void imprime() */



__device__ void imprimeVectorDevice (int *vector)
{
    printf("%d "),vector[blockIdx.x * blockDim.x + threadIdx.x];
}




__host__ void imprimeVectorHost (int *vector, int n)
{
    for (int i=0; i<n; i++)
    {
        printf("%d ",vector[i]);
    }
    printf("\n");
    
}



__global__ void sumaVectores (int*a, int*b, int*c , int n)
{
    // Se hace asi porque no se trabaja con un unico bloque, se tiene mas de un bloque
    int id_hilo= blockIdx.x * blockDim.x + threadIdx.x;

    if (id_hilo<n)
    {
        c[id_hilo]= a[id_hilo]+b[id_hilo];
        // Aqui se hara la prueba de ejecutar imprimeVector_Host desde la GPU, no deja al momento de compilar
        //imprimeVectorHost(c,n);
        imprimeVectorDevice(c);
    }
}


int main(int argc, char const *argv[])
{
    srand(time(NULL));

    // Parametro del main para el tamaño de los arreglos
    int n= atoi(argv[1]);

    // Arreglos dispositivo y host
    int *a_dispositivo, *b_dispositivo, *c_dispositivo;
    int *a_host= (int *)malloc(sizeof(int)*n);
    int *b_host= (int *)malloc(sizeof(int)*n);
    int *c_host= (int *)malloc(sizeof(int)*n);

    // Inicialziar arreglos
    for (int i=0; i<n; i++)
    {
        a_host[i]= 1+rand() % n;
        b_host[i]= 1+rand() % n;
        c_host[i]= 1+rand() % n;
    }
    

    // Declarar en GPU
    cudaMalloc(&a_dispositivo, n*sizeof(int));
    cudaMalloc(&b_dispositivo, n*sizeof(int));
    cudaMalloc(&c_dispositivo, n*sizeof(int));

    // Copiar del host al dispositivo
    cudaMemcpy(a_dispositivo, a_host, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dispositivo, b_host, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dispositivo, c_host, n*sizeof(int), cudaMemcpyHostToDevice);


    // Sacar las propiedades del dispositivo
    cudaDeviceProp propiedades; 
    cudaGetDeviceProperties (&propiedades,0);

    int tamanio_bloque= propiedades.maxThreadsPerBlock;
    int numero_bloques= (n*tamanio_bloque-1) / tamanio_bloque;


    sumaVectores<<<numero_bloques, tamanio_bloque>>>(a_dispositivo, b_dispositivo, c_dispositivo, n);
    cudaDeviceSynchronize();

    // Copiar del dispositivo al host
    cudaMemcpy(a_host, a_dispositivo, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host, b_dispositivo, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_host, c_dispositivo, n*sizeof(int), cudaMemcpyDeviceToHost);


    printf("\nLos resultados del host: ");
    imprimeVectorHost(c_host,n);

    // Liberar recursos del dispositivo
    cudaFree(a_dispositivo);
    cudaFree(b_dispositivo);
    cudaFree(c_dispositivo);


    return 0;
}

