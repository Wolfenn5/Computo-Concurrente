#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

/* Este programa trabaja con 
SISD Single Instruction Data
Y 
MIMD Multiple Instruction Data
*/



void * calculo (void * arg)
{
    int id= *(int *) arg;
    printf("\nSoy el hilo %d y estoy ejecutando mi instruccion", id);
    sleep(1);
    printf("\nSoy el hilo %d y termine mi instruccion\n",id);
    return NULL;
}



int main(int argc, char const *argv[])
{
    printf("\nVamos a simular un flujo unico de instruccionex (SISD)\n"); // Como no hay hilos, solo se crea un marco de ejecucion cuando se llama a la funcion
    for (int i=0; i<10; i++)
    {
        calculo(&i);
    }
    

    printf("\nVamos a simular multiples flujos de instrucciones (MIMD)\n");
    pthread_t hilos[10];
    for (int i=0; i <10; i++)
    {
        int * id= malloc(sizeof(int));
        * id= i;
        pthread_create(&hilos[i], NULL, calculo, (void *) id);
    }
    for (int i=0; i<10; i++)
    {
        pthread_join(hilos[i], NULL);
    }
    

    return 0;
}
