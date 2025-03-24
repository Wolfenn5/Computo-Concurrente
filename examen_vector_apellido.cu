#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
Apellido paterno: GARCIA
 La primer letra del apellido es: G (corresponde a la septima letra del abecedario)
 entonces 
 tam_vector= 7*500= 3500 elementos

 La ultima letra es: A (corresponde a la primer letra del abecedario)
 entonces 
 num_hilos_por_bloque= 1*4= 4

 

tambloque= 4 por lo anteriormente calculado y restriccion del problema seran 4 hilos solamente por bloque
entonces
numbloques= (N+tambloque-1) / tambloque    -->   donde tambloque=4 y N= 3500 
entonces
numbloques= (3500+4-1) / 4 = 875
*/

// Nota: Se hicieron pruebas por ejemplo con tam_vector de 10 (elementos) y se noto que si bien hacia el calculo, imprimia de forma incorrecta, asi que se determino que el programa solo funciona con el tamaño de vectores,numero de bloque y tamaño de bloque previamente calculados


__global__ void MultiplicarVectores (int * vectorA_dispositivo, int * vectorB_dispositivo, int * vectorC_dispositivo, int N) // N es el tam_vector
{
    int idHilo= (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (idHilo < N)
    {
        vectorC_dispositivo[idHilo]= vectorA_dispositivo[idHilo] * vectorB_dispositivo[idHilo]; // cada hilo recibe un id de la posicion que va a trabajar
    }
}




int main(int argc, char const *argv[])
{
    int tam_vector= 30; 
    // int tam_vector= 10; // tamaño de 10 (para probar)

    srand(time(NULL)); // para utilizar valores aleatorios 

    // Declaracion de vectores
    int *vectorA_dispositivo, *vectorB_dispositivo, *vectorC_dispositivo;
    int *vectorA_host, *vectorB_host, *vectorC_host;


    // Manejo de los arreglos en el host
    vectorA_host= (int *)malloc(tam_vector*sizeof(int));
    vectorB_host= (int *)malloc(tam_vector*sizeof(int));
    vectorC_host= (int *)malloc(tam_vector*sizeof(int));


    // inicializar arreglos del host con valores aleatorios
    for (int i=0; i<tam_vector; i++)
    {
        //vectorA_host[i]= 10 + rand() % 11; // valores aleatorios entre 0 y 10
        vectorA_host[i]= i+1; // valores sucesivos a partir de 1 (para probar)
    }
    for (int i=0; i<tam_vector; i++)
    {
        //vectorB_host[i]= 10 + rand() % 11; // valores aleatorios entre 0 y 10
        vectorB_host[i]= i+1; // valores sucesivos a partir de 1 (para probar)
    }
    for (int i=0; i<tam_vector; i++)
    {
        //vectorC_host[i]= 10 + rand() % 11; // valores aleatorios entre 0 y 10
        vectorC_host[i]= i+1; // valores sucesivos a partir de 1 (para probar)
    }


    // Declaracion de la memoria en el dispositivo (GPU)
    cudaMalloc(&vectorA_dispositivo, tam_vector*sizeof(int));
    cudaMalloc(&vectorB_dispositivo, tam_vector*sizeof(int));
    cudaMalloc(&vectorC_dispositivo, tam_vector*sizeof(int));


    // Copiar del host al dispositivo
    cudaMemcpy(vectorA_dispositivo, vectorA_host, tam_vector*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(vectorB_dispositivo, vectorB_host, tam_vector*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(vectorC_dispositivo, vectorC_host, tam_vector*sizeof(int),cudaMemcpyHostToDevice);
    
  
    // Tamaño del bloque y numero de bloques obtenidos del calculo previo (se encuentra debajo de las bibliotecas)
    int tamanio_bloque= 4;
    int num_bloques= 875; 
    

    // Medir tiempo del dispositivo (GPU) usando eventos de cuda
    cudaEvent_t inicio, fin; // vendria siendo el equivalente a clock_t 
    // se declaran variables que van a ser eventos
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);



    cudaEventRecord(inicio); // se marca en donde va a empezar a medir el tiempo de GPU que es cuando se lanza el kernel para empezar a hacer calculos
    // Lanzar el kernel
    MultiplicarVectores<<<num_bloques,tamanio_bloque>>>(vectorA_dispositivo, vectorB_dispositivo, vectorC_dispositivo, tam_vector); // kernel que va a trabajar
    cudaEventRecord(fin); // se marca en donde va a terminar de medirse el tiempo de GPU que es cuando el kernel ya acabo


    cudaDeviceSynchronize(); // es el equivalente a .join en hilos. Es importante utilizarlo para no trabar la GPU


    // Copiar del dispositivo al host
    cudaMemcpy(vectorC_host, vectorC_dispositivo, tam_vector*sizeof(int),cudaMemcpyDeviceToHost);



    printf("\nLos primeros 5 elementos son:\n");
    for (int i=0; i<5; i++)
    {
        printf("%d ", vectorC_host[i]);
    }
    printf("\n");



    printf("\nLos ultimos 5 elementos son:\n");
    for (int i=tam_vector-5; i<tam_vector; i++)
    {
        printf("%d ", vectorC_host[i]);
    }
    printf("\n");


    // Calcular el tiempo que le tomo a la GPU hacer los calculos
    float tiempo_GPU=0; 
    cudaEventElapsedTime(&tiempo_GPU, inicio, fin); // se indica en donde se va a guardar, el inicio y el final. Siempre va a devolver el tiempo en milisegundos
    printf("\nEl tiempo de ejecucion del dispositivo (GPU) fue de: %f segundos\n",tiempo_GPU/1000); // se divide tiempo/1000 para dar el tiempo en segundos en vez de milisegundos



    // Liberar memoria del dispositivo (GPU)
    cudaFree(vectorA_dispositivo);
    cudaFree(vectorB_dispositivo);
    cudaFree(vectorC_dispositivo);
    

    // Liberar memoria del host (CPU)
    free(vectorA_host);
    free(vectorB_host);
    free(vectorC_host);
   

    return 0;
}
