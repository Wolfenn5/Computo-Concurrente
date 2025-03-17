#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void sumaVectores (int * A_dispositivo, int * B_dispositivo, int * C_dispositivo, int N) // N es la dimension
{
    int idHilo= (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (idHilo < N)
    {
        //printf("\n%d+%d ", A_dispositivo[idHilo], B_dispositivo[idHilo]);
        C_dispositivo[idHilo]= A_dispositivo[idHilo] + B_dispositivo[idHilo]; // cada hilo recibe un id de la posicion que va a trabajar
    }
}




int main(int argc, char const *argv[])
{
    srand(time(NULL));
    int dimension= atoi (argv[1]); // tamaño de los arreglos desde el main


    int *A_dispositivo, *B_dispositivo, *C_dispositivo;
    int *A_host, *B_host, *C_host;

    // Manejo de los arreglos en el host
    A_host= (int *)malloc(dimension*sizeof(int));
    B_host= (int *)malloc(dimension*sizeof(int));
    C_host= (int *)malloc(dimension*sizeof(int));

    // inicializar arreglos del host
    for (int i=0; i<dimension; i++)
    {
        A_host[i]= 10 + rand() % 90;
        B_host[i]= 10 + rand() % 90;
        C_host[i]= 10 + rand() % 90;
    }

    // Declaracion de la memoria en el dispositivo (GPU)
    cudaMalloc(&A_dispositivo, dimension*sizeof(int));
    cudaMalloc(&B_dispositivo, dimension*sizeof(int));
    cudaMalloc(&C_dispositivo, dimension*sizeof(int));


    //Mover la memoria del host al dispositivo
    cudaMemcpy(A_dispositivo, A_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(B_dispositivo, B_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(C_dispositivo, C_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
    
    // Sacar las propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);
    int tamanio_bloque= propiedades.maxThreadsPerBlock; // saber el tamaño del bloque


    // Formula para dividir cualquier vector o arreglo para trabajar con cuda
    int num_bloques= (dimension+tamanio_bloque-1)/tamanio_bloque; // saber cuantos bloques necesitamos para trabajar
    

    sumaVectores<<<num_bloques,tamanio_bloque>>>(A_dispositivo, B_dispositivo, C_dispositivo, dimension); // kernel que va a trabajar



    // Regresar la informacion del dispositivo al host
    cudaMemcpy(C_host, C_dispositivo, dimension*sizeof(int),cudaMemcpyDeviceToHost);



    printf("\nEl arreglo C es:\n");
    for (int i=0; i<dimension; i++)
    {
        printf("%d ", C_host[i]);
    }
    printf("\n");

    // Liberar los recursos de la GPU
    cudaFree(A_dispositivo);
    cudaFree(B_dispositivo);
    cudaFree(C_dispositivo);
    



    return 0;
}
