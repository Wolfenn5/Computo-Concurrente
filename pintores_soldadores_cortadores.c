#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <semaphore.h>
#include <stdlib.h>

/* Productor consumidor de cortadores

Los cortadores no pueden comenzar con un nuevo producto hasta que los pintores hayan terminado el anterior
Los soldadores deben esperar a que los cortadores finalicen antes de comenzar a ensamblar
Los pintores deben esperar a que los soldadores terminen antes de iniciar la pintura
Se deben ensamblar un total de M productos en paralelo, con multiples trabajadores en cada etapa

En una línea de ensamblaje de productos, existen tres tipos de trabajadores:
Cortadores: Se encargan de cortar las piezas necesarias para el ensamblaje.
Soldadores: Ensamblan las piezas cortadas mediante soldadura.
Pintores: Pintan el producto ensamblado para su acabado final.

*/

/* Semaforos, barreras, variables de condicion*/


pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER; // mutex

pthread_cond_t condicion_cortadores= PTHREAD_COND_INITIALIZER; // variable
pthread_cond_t condicion_soldadores= PTHREAD_COND_INITIALIZER; // variable
pthread_cond_t condicion_pintores= PTHREAD_COND_INITIALIZER; // variable

sem_t semaforo_pintores;
sem_t semaforo_cortadores;
sem_t semaforo_soldadores;


pthread_barrier_t barrera; 

int M;  // Mproductos como variable global



void * cortadores (void * arg)
{
  int id_cortadores= *(int *)arg; // se recibe el numero de cortadores

  for (int i=0; i<id_cortadores; i++)
  {
    //sem_wait(&semaforo_pintores); // esperar a que los pintores terminen de pintar
    sem_wait(&semaforo_cortadores);

    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&condicion_pintores, &mutex); // esperar a que los pintores terminen de pintar el producto anterior
    printf("\nEl cortador: %d esta cortando...",i);
    pthread_cond_signal(&condicion_soldadores); // avisar a los soldadores que ya se corto el material
    pthread_mutex_unlock(&mutex);
    sleep(2); // simular el tiempo que esta cortando


    sem_post(&semaforo_soldadores); // indicarle a los soldadores que ya se corto material
  }
  pthread_exit(NULL);
}



void * soldadores (void *arg)
{
  int id_soldadores= *(int *)arg; // se recibe el numero de soldadores
  for (int i=0; i<id_soldadores; i++)
  {
    //sem_wait(&semaforo_cortadores); // esperar a que los cortadores terminen
    sem_wait(&semaforo_soldadores);

    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&condicion_cortadores, &mutex); // esperar a que los cortadores terminen de cortar el material
    printf("\nEl soldador: %d esta soldando...",i);
    pthread_cond_signal(&condicion_pintores); // avisar a los pintores que ya se termino
    pthread_mutex_unlock(&mutex);
    sleep(2); // simular el tiempo que esta cortando

    sem_post(&semaforo_pintores); // indicar a los pintores que los soldadores ya terminaron
  }
  pthread_exit(NULL);
}



void * pintores (void * arg)
{
  int id_pintores= *(int *)arg; // se recibe el numero de pintores
  for (int i=0; i<id_pintores; i++)
  {
    //sem_wait(&semaforo_soldadores); // esperar a que los soldadores terminen
    sem_wait(&semaforo_pintores);

    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&condicion_soldadores, &mutex); // esperar a que los soldadores corten el material
    printf("\nEl pintor: %d esta pintando...",i);
    pthread_cond_signal(&condicion_cortadores); // avisar a los cortadores
    pthread_mutex_unlock(&mutex);
    sleep(2); // simular el tiempo que esta cortando

    pthread_barrier_wait(&barrera); // esperar a que todos los cortadores terminen para poder empezar a pintar

    //sem_post(&semaforo_cortadores); // indicar a los cortadores que todos los pintores ya terminaron
  }
  pthread_exit(NULL);
}





int main(int argc, char const *argv[])
{
  // Parametros de main
  // 3 argumentos:
  // --> nombre_programa Mproductos cortadores soldadores pintores
  M= atoi(argv[1]); 
  int num_cortadores= atoi(argv[2]); 
  int num_soldadores= atoi(argv[3]); 
  int num_pintores= atoi(argv[4]); 
  


  // Declarar hilos
  pthread_t hilo_cortadores[num_cortadores], hilo_soldadores[num_soldadores], hilo_pintores[num_pintores];


  // inicializar la barrera
  pthread_barrier_init(&barrera,NULL,num_cortadores); // argumentos: barrera, atributos predeterminados, contador del numero de hilos que debemos esperar en la barrera antes de continuar
  // se pone el numero de cortadores porque los pintores deben esperar a que todos los cortadores terminen antes de empezar a pintar


  // Inicializacion de semaforos
  // se inicializa en 0 
  sem_init(&semaforo_cortadores, 0, num_cortadores);
  sem_init(&semaforo_soldadores, 0, num_soldadores);
  sem_init(&semaforo_pintores, 0, num_pintores);

  // Creacion de Hilos
  //Cortadores
  for(int i=0; i<num_cortadores; i++)
  {
    int * id = malloc(sizeof(int));//cada cortador tiene su identificador "unico"
    * id = i;
    pthread_create(&hilo_cortadores[i],NULL, cortadores,(void *)id);
  }

  //Soldadores
  for(int i=0; i<num_soldadores; i++)
  {
    int * id = malloc(sizeof(int));//cada soldador tiene su identificador "unico"
    * id = i;
    pthread_create(&hilo_soldadores[i],NULL, soldadores,(void *)id);
  }

  //Pintores
  for(int i=0; i<num_pintores; i++)
  {
    int * id = malloc(sizeof(int));//cada pintor tiene su identificador "unico"
    * id = i;
    pthread_create(&hilo_pintores[i],NULL, pintores,(void *)id);
  }
  


  //Destruccion de hilos
  // Cortadores
  for(int i=0; i<num_cortadores; i++)
  {
    pthread_join(hilo_cortadores[i],NULL);
  }
  // Soldadores
  for(int i=0; i<num_soldadores; i++)
  {
    pthread_join(hilo_soldadores[i],NULL);
  }
  // Pintores
  for(int i=0; i<num_pintores; i++)
  {
    pthread_join(hilo_pintores[i],NULL);
  }


  // Destruccion
  // Barrera
  pthread_barrier_destroy(&barrera);

  // Semaforo
  sem_destroy(&semaforo_cortadores);
  sem_destroy(&semaforo_soldadores);
  sem_destroy(&semaforo_pintores);

  // Mutex
  pthread_mutex_destroy(&mutex);

  return 0;
}
