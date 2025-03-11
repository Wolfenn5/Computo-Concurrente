#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

/* tamaños del vector 
1000
10000
100000
1000000
10000000
*/


typedef struct 
{
    int *A; // vecotr A
    int *B; // vecotr B
    int inicio; // inicio del vector
    int fin; // final del vector
    int id; // id del hilo
    long long resultado_parcial; // resultado parcial (que cada hilo hace) del producto punto 
} parametros;



void* producto_punto(void *arg) 
{
    parametros * argumentos= (parametros *)arg; // casting
    printf("\nSoy el hilo %d y voy a comenzar a calcular\n",argumentos->id);
    argumentos->resultado_parcial=0; // como se usa memoria dinamica y el vector sera de gran tamaño para las pruebas, se inicializa en 0 por si hay basura

    for(int i=argumentos->inicio; i<argumentos->fin; i++) 
    {
        argumentos->resultado_parcial+= ((long long)argumentos->A[i]) * (argumentos->B[i]); // se hace el producto punto y se almacena en resultado parcial porque cada hilo esta haciendo su parte que le toca
    }
    printf("\nSoy el hilo %d y termine de calcular\n",argumentos->id);
    pthread_exit(NULL);
}


int main(int argc, char const *argv[]) 
{
    /* Esta seccion se uso para saber como maneja los datos la computadora*/
	// printf("Tamano de int: %lu bytes\n", sizeof(int));                //4
    // printf("Tamano de long: %lu bytes\n", sizeof(long));              //4
    // printf("Tamano de long long: %lu bytes\n", sizeof(long long));    //8
    // printf("Tamano de float: %lu bytes\n", sizeof(float));            //4
    // printf("Tamano de double: %lu bytes\n", sizeof(double));          //8
    // printf("Tamano de char: %lu bytes\n", sizeof(char));              //1


    // Argumentos de main 
    // --> nombre tamaño del vector (A y B), numero de hilos
    int tamanio_vector=atoi(argv[1]);
    int num_hilos=atoi(argv[2]); 


    int tamanio_bloque=tamanio_vector/num_hilos; // numero de elementos del vector que cada hilo va a trabajar
    long long resultado_final=0; // resultado final de todo el producto punto

    // Vectores con memoria dinamica
    int *A= (int *)malloc(sizeof(int)* tamanio_vector);
    int *B= (int *)malloc(sizeof(int)* tamanio_vector);


    // Llenado de vectores A sera ascendente y B descendente
    for(int i=0; i<tamanio_vector; i++)
    {
        A[i]= i+1; // 1,2,3,4,5 ...
        B[i]= tamanio_vector-i; // ... 5,4,3,2,1
    }
    // Llenado de vectores (prueba con valores de la practica)
    // for(int i=0; i<5; i++)
    // {
    //     A[i]= i+1; 
    //     B[i]= i+6; 
    // }
    

    parametros argumentos[num_hilos]; // arreglo de argumentos para cada hilo
    pthread_t hilos[num_hilos]; // hilos


    // Creacion de hilos
    for(int i=0; i<num_hilos; i++)
    {
        argumentos[i].A= A;
        argumentos[i].B= B;
        argumentos[i].inicio= i*tamanio_bloque;

        if(i == num_hilos-1) // si es el ultimo hilo
        {
            argumentos[i].fin= tamanio_vector;
        }
        else
        {
            argumentos[i].fin= (i+1) * tamanio_bloque;
        }

        argumentos[i].id= i+1;
        argumentos[i].resultado_parcial=0;
        pthread_create(&hilos[i], NULL, producto_punto, (void *) &argumentos[i]);
    }


    // Esperar a que los hilos terminen
    for(int i=0; i<num_hilos; i++)
    {
        pthread_join(hilos[i],NULL);
        resultado_final+= argumentos[i].resultado_parcial;
    }

    printf("\n\nEl producto punto de A y B es: %lld\n",resultado_final);

    free(A);
    free(B);

    return 0;
}
