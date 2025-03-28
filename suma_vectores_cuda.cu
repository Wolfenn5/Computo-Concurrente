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


    //int tamanio_bloque= 512; // para el ejercicio de medir tiempos de ejecucion


    // Formula para dividir cualquier vector o arreglo para trabajar con cuda
    int num_bloques= (dimension+tamanio_bloque-1)/tamanio_bloque; // saber cuantos bloques necesitamos para trabajar

    

    cudaEvent_t inicio, fin; // vendria siendo el equivalente a clock_t 
    // se declaran variables que van a ser eventos
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);



    cudaEventRecord(inicio); // se marca en donde va a empezar a medir el tiempo de GPU que es cuando se lanza el kernel para empezar a hacer calculos
    sumaVectores<<<num_bloques,tamanio_bloque>>>(A_dispositivo, B_dispositivo, C_dispositivo, dimension); // kernel que va a trabajar
    cudaEventRecord(fin); // se marca en donde va a terminar de medirse el tiempo de GPU que es cuando el kernel ya acabo


    // Esperar a que los hilos de la GPU terminen
    cudaDeviceSynchronize(); // es el equivalente a .join en hilos. Es importante utilizarlo para no trabar la GPU


    // Regresar la informacion del dispositivo al host
    cudaMemcpy(C_host, C_dispositivo, dimension*sizeof(int),cudaMemcpyDeviceToHost);



    // Calcular el tiempo que le tomo a la GPU hacer los calculos
    float tiempo_GPU=0; 
    cudaEventElapsedTime(&tiempo_GPU, inicio, fin); // se indica en donde se va a guardar, el inicio y el final. Siempre va a devolver el tiempo en milisegundos
    printf("\nEl tiempo de ejecucion del dispositivo (GPU) fue de: %f segundos\n",tiempo_GPU/1000); // se divide tiempo/1000 para dar el tiempo en segundos en vez de milisegundos



    // printf("\nEl arreglo C es:\n");
    // for (int i=0; i<dimension; i++)
    // {
    //     printf("%d ", C_host[i]);
    // }
    // printf("\n");


    // Liberar los recursos de la GPU
    cudaFree(A_dispositivo);
    cudaFree(B_dispositivo);
    cudaFree(C_dispositivo);
    



    return 0;
}
