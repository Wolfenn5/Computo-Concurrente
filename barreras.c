#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUMHILOS 4

pthread_barrier_t barrera;

void * fases(void * arg)
{
    int id= *(int*)arg;
    // primera fase
    printf("\nHilo %d en la fase 1...\n", id);
    usleep(rand() % 1000000);
    printf("\nHilo %d termino la fase 1\n", id);

    //sincronizar con la barrera
    printf("\nHilo %d esperando a que todos terminen la fase 1\n",id);
    pthread_barrier_wait(&barrera);


    // segunda fase
    printf("\nHilo %d en la fase 2...\n", id);
    usleep(rand() % 1000000);
    printf("\nHilo %d termino la fase 2\n", id);
    pthread_exit(NULL);
}



int main(int argc, char const *argv[])
{
    pthread_t hilos[NUMHILOS];

    srand(time(NULL));
    
    // inicializar la barrera
    pthread_barrier_init(&barrera,NULL,NUMHILOS); // argumentos: barrera, atributos predeterminados, contador del numero de hilos que debemos esperar en la barrera antes de continuar


    // Crear hilos
    for (int i=0; i<NUMHILOS; i++)
    {
        int * id= malloc(sizeof(int));
        * id=i;
        pthread_create(&hilos[i], NULL, fases, (void *) id);
    }
    
    // Esperar hilos
    for (int i=0; i<NUMHILOS; i++)
    {
        pthread_join(hilos[i],NULL);
    }
    
    // Destruir la barrera
    pthread_barrier_destroy(&barrera);

    return 0;
}
