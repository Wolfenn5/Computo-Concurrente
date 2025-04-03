#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
/*Calcular el numero de bloques necesario para cubrir 1 millon de elementos con un tamaño de bloque de 1024 hilos
int N= 1000000000;
int tambloque= 1024;
int numbloques= (N+tambloque-1) / tambloque;
La operacion seria (1,000,000,000 + 1024 -1) / 1024 = 976,567 numbloques

Calcular el tamaño de la malla "tamanoMalla" utilizando el numero de bloques anterior de 976,567
La operacion seria (1,000,000,000 + 976,567 -1) / 976,567 = 1024.99 = tamanoMalla= (1025,1025)

Calcular los hilos totalHilos multiplicando el tamaño de la malla en cada dimension por el tamaño del bloque
totalHilos= 1025* 1025 * 1024= 1,075,840,000;
*/


__global__ void SumaMatrices(float *matrizA, float *matrizB, float *matrizC, int n) 
{
    // la fila va a servir como id del hilo como en el problema de vector_matriz
    // int id_hilo= blockIdx.x * blockDim.x + threadIdx.x; 


    int fila= blockIdx.x * blockDim.x + threadIdx.x; // id bloque, dimension bloque, id hilo en ese bloque ; bloque 0, dimension bloque 512 + id hilo 0 seria el hilo 512
    int columna= blockIdx.y * blockDim.y + threadIdx.y; // lo mismo que el anterior pero como eje y

    if (fila<n && columna<n)  // si el id hilo en eje x ó y (fila, columna) es menor al numero total de datos (N) hacer el procesamiento
    {
        float suma= 0.0; // inicializar el elemento correspondiente al "arreglo" del resultado


        // calcular el indice porque la matriz se esta trabajando como un arreglo (doble apuntador)
        suma= matrizA[fila*n+columna] + matrizB[fila*n+columna];
        matrizC[fila*n+columna]= suma;
    }
}



int main(int argc, char const *argv[]) 
{
    // Parametro del main para el tamaño de la matriz NxN
    int N= atoi(argv[1]);
    srand(time(NULL));


    // Declarar matrices en el host (CPU)
    float *matrizA_host= (float*) malloc(sizeof(float)*N*N);
    float *matrizB_host= (float*) malloc(sizeof(float)*N*N);
    float *matrizC_host= (float*) malloc(sizeof(float)*N*N);


    // Declarar matrices en el dispositivo (GPU)
    float *matrizA_dispositivo, *matrizB_dispositivo, *matrizC_dispositivo;
    cudaMalloc(&matrizA_dispositivo, N*N* sizeof(float));
    cudaMalloc(&matrizB_dispositivo, N*N* sizeof(float));
    cudaMalloc(&matrizC_dispositivo, N*N* sizeof(float));



    // Llenado de matrices
    for (int i=0; i<N*N; i++) 
    {
        matrizA_host[i]= (float) rand()/RAND_MAX; // valores entre 0 y 1
        //matrizA_host[i]= i+1; // valores consecutivos a partir de 1 (para probar)
    }

    for (int i=0; i<N*N; i++) 
    {
        matrizB_host[i]= (float) rand()/RAND_MAX; // valores entre 0 y 1
        //matrizB_host[i]= i+1; // valores consecutivos a partir de 1 (para probar)
    }


    // Imprimir matriz A
    // printf("\nMatriz A:\n");
    // for (int i=0; i<N; i++) 
    // {
    //     for (int j=0; j<N; j++) 
    //     {
    //         printf("%f ", matrizA_host[i*N+j]);
    //     }
    //     printf("\n");
    // }

    // Imprimir matriz B
    // printf("\nMatriz B:\n");
    // for (int i=0; i<N; i++) 
    // {
    //     for (int j=0; j<N; j++) 
    //     {
    //         printf("%f ", matrizB_host[i*N+j]);
    //     }
    //     printf("\n");
    // }


    // Copiar del host al dispositivo
    cudaMemcpy(matrizA_dispositivo, matrizA_host, sizeof(float) *N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(matrizB_dispositivo, matrizB_host, sizeof(float) *N*N, cudaMemcpyHostToDevice);


    // Calcular el numero de hilos a ocupar
    // Sacar propiedades del dispositivo
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);
    int tam_bloque= propiedades.maxThreadsPerBlock; // calcular tamaño optimo del bloque
    // int tam_bloque= 1024; // para el ejercicio de medir tiempos de ejecucion
    int num_bloques= (N+tam_bloque-1) / tam_bloque; // calcular el numero de bloques ; N es el numero de datos (en este caso el tamaño de la matriz), la formula es universal
    // Si se quiere saber el numero de hilos maximo se multiplica tam_bloque*numero_bloques

    // Definir el tamaño de la malla y el bloque
    dim3 tamanio_bloque(num_bloques,num_bloques);
    dim3 tamanio_malla((N+num_bloques-1) / num_bloques,(N+num_bloques-1) / num_bloques);



    // Medir tiempo del dispositivo (GPU) usando eventos de cuda
    cudaEvent_t inicio, fin; // vendria siendo el equivalente a clock_t 
    // se declaran variables que van a ser eventos
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);



    cudaEventRecord(inicio); // se marca en donde va a empezar a medir el tiempo de GPU que es cuando se lanza el kernel para empezar a hacer calculos
    // Lanzar el kernel
    SumaMatrices<<<tamanio_malla, tamanio_bloque>>>(matrizA_dispositivo, matrizB_dispositivo, matrizC_dispositivo, N);

    
    // Esperar a que los hilos de la GPU terminen
    cudaDeviceSynchronize(); // es el equivalente a .join en hilos. Es importante utilizarlo para no trabar la GPU

  
    // Copiar del dispositivo al host
    cudaMemcpy(matrizC_host, matrizC_dispositivo, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    cudaEventRecord(fin); // se marca en donde va a terminar de medirse el tiempo de GPU que es cuando el kernel ya acabo que es despues de que se termine la transferencia de datos del dispositivo la host



    // Imprimir matriz C
    printf("\nMatriz resultado:\n");
    // for (int i=0; i<N; i++) 
    // {
    //     for (int j=0; j<N; j++) 
    //     {
    //         printf("%f ", matrizC_host[i*N+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n%f\n", matrizC_host[(N*N)-1]); // ultimo elemento nadamas (para probar con valores grandes cuando se hace con numeros consecutivos)


    // Calcular el tiempo que le tomo a la GPU hacer los calculos
    float tiempo_GPU=0; 
    cudaEventElapsedTime(&tiempo_GPU, inicio, fin); // se indica en donde se va a guardar, el inicio y el final. Siempre va a devolver el tiempo en milisegundos
    printf("\nEl tiempo de ejecucion del dispositivo (GPU) fue de: %f segundos\n",tiempo_GPU/1000); // se divide tiempo/1000 para dar el tiempo en segundos en vez de milisegundos

    // Liberar memoria del dispositivo (GPU)
    cudaFree(matrizA_dispositivo);
    cudaFree(matrizB_dispositivo);
    cudaFree(matrizC_dispositivo);


    // Liberar memoria del host (CPU)
    free(matrizA_host);
    free(matrizB_host);
    free(matrizC_host);

    return 0;
}
