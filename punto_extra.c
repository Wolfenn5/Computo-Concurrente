/* Productor consumidor de cortadores
Los cortadores no pueden comenzar con un nuevo producto hasta que los pintores hayan terminado el anterior
Los soldadores deben esperar a que los cortadores finalicen antes de comenzar a ensamblar
Los pintores deben esperar a que los soldadores terminen antes de iniciar la pintura
Se deben ensamblar un total de M productos en paralelo, con multiples trabajadores en cada etapa
*/

/* Semaforos, barreras, variables de condicion*/

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <semaphore.h>

#define BUFFER_SIZE 30
int buffer[BUFFER_SIZE];


pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER; // mutex
pthread_cond_t condicion_cortadores= PTHREAD_COND_INITIALIZER; // variable
pthread_cond_t condicion_soldadores= PTHREAD_COND_INITIALIZER; // variable
pthread_cond_t condicion_pintores= PTHREAD_COND_INITIALIZER; // variable

sem_t semaforo;


pthread_barrier_t barrera; 

/*
pthread_barrier_init(&barrera);
pthread_barrier_wait(&barrera);
pthread_barrier_destroy(&barrera);
*/


void * cortadores (void * arg)
{
    
}


void * soldadores (void *arg)
{

}


void * pintores (void * arg)
{

}





int main(int argc, char const *argv[])
{
    // Parametros de main
    // 3 argumentos:
    // --> nombre Mproductos cortadores soldadores pintores
    int M= atoi(argv[1]); 
    int num_cortadores= atoi(argv[2]); 
    int num_soldadores= atoi(argv[3]); 
    int num_pintores= atoi(argv[4]); 
    


    // Declarar hilos
    pthread_t hilo_cortadores[num_cortadores], hilo_soldadores[num_soldadores], hilo_pintores[num_pintores];


    // inicializar la barrera
    pthread_barrier_init(&barrera,NULL,num_cortadores); // argumentos: barrera, atributos predeterminados, contador del numero de hilos que debemos esperar en la barrera antes de continuar



    // Hilos
    //Cortadores
	for(int i=0;i<num_cortadores;++i)
    {
		int * id = malloc(sizeof(int));//cada cortador tiene su identificador "unico"
		* id = i;
		pthread_create(&hilo_cortadores[i],NULL, cortadores,(void *)id);
	}
	// Esperar que terminen los hilos
	for(int i=0;i<num_cortadores;++i)
    {
		pthread_join(hilo_cortadores[i],NULL);
    }

    //Soldadores
	for(int i=0;i<num_soldadores;++i){
		int * id = malloc(sizeof(int));//cada cortador tiene su identificador "unico"
		* id = i;
		pthread_create(&hilo_soldadores[i],NULL, cortadores,(void *)id);
	}
	// Esperar que terminen los hilos
	for(int i=0;i<num_soldadores;++i)
    {
		pthread_join(hilo_soldadores[i],NULL);
    }

    //Pintores
	for(int i=0;i<num_pintores;++i){
		int * id = malloc(sizeof(int));//cada cortador tiene su identificador "unico"
		* id = i;
		pthread_create(&hilo_pintores[i],NULL, cortadores,(void *)id);
	}
	// Esperar que terminen los hilos
	for(int i=0;i<num_pintores;++i)
    {
		pthread_join(hilo_pintores[i],NULL);
    }





    // Destruir la barera
    pthread_barrier_destroy(&barrera);

    // Destruir el semaforo
    sem_destroy(&semaforo);

    return 0;
}
