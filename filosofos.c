#include <stdio.h>
#include <pthread.h> // para hilos como pthread_Create y pthread_join
#include <stdlib.h> // para atoi para argumentos del main
#include <semaphore.h> // para semaforos sem_wait sem_post
#include <unistd.h> // para sleep

/* El programa se ejecutara siempre por la condicion de los whiles con el fin de ver como cambian los filosofos que comen y piensan con cada ejecucion*/

#define Filosofos 5 // macro para el numero de filosofos



// Arreglo de semaforos, en si cada palillo va a ser un semaforo
sem_t palillos[Filosofos]; 
// --> palillos[palillo0, palillo1, palillo2, palillo3, palillo4]



void *filosofo(void *arg) 
{
    int id= *(int *)arg; // se recibe el id del filosofo

    while (1) 
    {
        
        /* ---------- Pensar ----------*/
        printf("Yo el filosofo %d estoy pensando\n", id);
        sleep(4); // se simula que el filosofo piensa por 4 segundos



        // Tomar los palillos (hacer que el semaforo no deje pasar o mejor dicho bloquee el palillo y que no este disponible)     
        // --> decrementa el valor del contador del semaforo poniendolo en 0 simulando que el palillo no esta disponible
        sem_wait(&palillos[id]); // palillo de la izquierda
        sem_wait(&palillos[(id + 1) % Filosofos]); // paillo de la derecha, si el filoso 4 quiere agarrar un paillo con % hace que el palillo a tomar sea el palillo 0


        /* ---------- Comer ----------*/
        printf("Yo el filosofo %d estoy comiendo...\n", id);
        printf("Ya termine de comer\n");
        sleep(4); // se simla que el filosofo come por 4 segundos



        // Soltar los palillos (hacer que el semaforo deje pasar o mejor dicho libere el palillo y este disponible)       
        // --> incrementa el valor del contador del semaforo poniendolo en 1 simulando que el palillo esta disponible
        sem_post(&palillos[id]); // palillo de la izquierda
        sem_post(&palillos[(id + 1) % Filosofos]); // paillo de la derecha, si el filoso 4 quiere agarrar un paillo con % hace que el palillo a tomar sea el palillo 0
    }

    return NULL;
}




int main(int argc, char *argv[]) 
{
    pthread_t filosofos[Filosofos]; // hilos que seran filosofos


    int ids[Filosofos];

    // Inicializacion de semaforos (palillos) 
    for (int i=0; i<Filosofos; i++) 
    {
        sem_init(&palillos[i], 0, 1); // se inicializa con 0 porque cada palillo sera tomado por un filosofo (hilo) y el contador en 1 indicando que el palillo[i] esta disponible desde el inicio
    }



    // Creacion de hilos (filosofos)
    for (int i=0; i<Filosofos; i++) 
    {
        ids[i]= i;
        pthread_create(&filosofos[i], NULL, filosofo, (void *)&ids[i]); // argumentos: hilo, NULL, funcion a hacer, id del filosofo
    }


    // Esperar a que los hilos terminen
    for (int i=0; i<Filosofos; i++)
    {
        pthread_join(filosofos[i], NULL); // argumentos: hilo, NULL
    }



    // Destruir los semáforos (palillos)
    for (int i = 0; i < Filosofos; i++) 
    {
        sem_destroy(&palillos[i]);
    }



    return 0;
}
