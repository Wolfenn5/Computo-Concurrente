#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct 
{
    float * A;
    float * B;
    int inicio;
    int fin;
    int id;
} parametros;


void * suma_vectores (void * arg)
{
    parametros * argumentos= (parametros *)arg; // casting

    printf("\nSoy el hilo %d y voy a comenzar a sumar mis elementos\n", argumentos->id);
    for (int i=argumentos->inicio; i<argumentos->fin; i++) // el operador -> se usa porque "argumentos" es un apuntador
    {
        argumentos->A[i]+= argumentos->B[i]; // se hace la suma y se almacena en el vector A
    }
    printf("\nSoy el hilo %d y termine de sumar mis elementos\n", argumentos->id);
    pthread_exit(NULL);
}

/*Este programa hace la suma de los elementos de 2 vectores con taxonomia de MIMD*/
// Ejecutan los nucleos mediante los hilos un conjunto de instrucciones en diferentes datos de forma simultanea (concurrente)


int main(int argc, char const *argv[])
{
    srand(time(NULL));
    // Parametros de main
    // --> nombre, suma, numero de hilos
    int N= atoi(argv[1]);
    int num_hilos= atoi (argv[2]);
    int tamanio_bloque= N / num_hilos; // cada bloque es del tamaño correspondiente a la proporcion que le toca a cada hilo
    // hay que verificar que los valores sean divididos de tal manera que no se pierda informacion

    // Arreglos con memoria dinamica
    float *A, *B; 
    A= (float *)malloc(sizeof(float)*N);
    B= (float *)malloc(sizeof(float)*N);


    // Arreglo de argumentos para cada hilo
    parametros argumentos [num_hilos];


    // Inicializacion de vectores
    for (int i=0; i<N; i++)
    {
        A[i]= (float)rand()/RAND_MAX; // valores entre 0 y 1
        B[i]= (float)rand()/RAND_MAX;
    }
    // se van a guardar los resultados en el vector A



    printf("\nEl arreglo A\n");
    for (int i=0; i<N; i++)
    {
        printf("%f ", A[i]);
    }
    printf("\n");


    printf("\nEl arreglo B\n");
    for (int i=0; i<N; i++)
    {
        printf("%f ", B[i]);
    }
    printf("\n");

    // Declarar hilos
    pthread_t hilos[num_hilos];

    // Crear hilos
    for (int i=0; i<num_hilos; i++)
    {
        argumentos[i].A= A;
        argumentos[i].B= B;
        argumentos[i].inicio= i* tamanio_bloque;
        argumentos[i].fin= (i+1) * tamanio_bloque;
        argumentos[i].id = i+1;
        pthread_create(&hilos[i], NULL, suma_vectores, (void*) &argumentos[i]);

        if (i == num_hilos-1) // si es el ultimo hilo
        {
            argumentos[i].fin= N;
        }
        
    }


    // Esperar hilos
    for (int i=0; i<num_hilos; i++)
    {
        pthread_join(hilos[i],NULL);
    }
    
    printf("\nEl arreglo resultante es:\n");

    for (int i=0; i<N; i++)
    {
        printf("%f ", A[i]);
    }
    printf("\n");

    // Opcional si es linux, pero si es windows y mac si es obligatorio
    free(A);
    free(B);
    

    return 0;
}





