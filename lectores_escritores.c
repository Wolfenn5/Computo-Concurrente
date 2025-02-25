#include <stdio.h>
#include <pthread.h> // para hilos como pthread_Create y pthread_join
#include <stdlib.h> // para atoi para argumentos del main
#include <semaphore.h> // para semaforos sem_wait sem_post
#include <unistd.h> // para sleep

/* El programa se ejecutara siempre por la condicion de los whiles con el fin de ver como cambian los lectores y escritores con cada ejecucion*/

int libro= 0; // el contenido del libro solo tendra el id del escritor que escriba en el 


// Declaracion de mutex y semaforo
pthread_mutex_t mutex; // mutex para proteger a los lectores
sem_t semaforo; // controla el acceso exclusivo al libro


void *lector(void *arg) 
{
    int id=*(int *)arg; // se recibe el id del lector

    while(1) 
    {
        printf("Lector %d esta leyendo el libro con contenido: %d",id, libro);
        printf("\n");
        sleep(2); // se simula que los lectores leen cada 2 segundos
    }
    return NULL;
}




void *escritor(void *arg) 
{
    int id=*(int *)arg; // se recibe el id del escritor
    while(1) 
    {

        sem_wait(&semaforo); // se activa el semaforo para que el escritor pueda escriba en el libro y no se pueda leer
        printf("Yo el escritor %d estoy escribiendo en el libro...\n",id);


        pthread_mutex_lock(&mutex); // el candado protege la variable para que solamente el escritor actual pueda acceder a ella (libro)
        libro=id; // el escritor escribe su id en el libro
        pthread_mutex_unlock(&mutex); // se libera el candado


        printf("Ya termine de escribir\n");
        sleep(2); // se simula que el escritor tarda 2 segundos en escribir
        sem_post(&semaforo); // se quita el semaforo cuando el escritor ya escribio en el libro y deja pasar a los lectores para que lean


        sleep(3); // tiempo de 3 segundos para que el escritor que acaba de escribir pueda volver a escribir
    }
    return NULL;
}



int main(int argc,char const *argv[]) 
{
    // Parametros del main, argv[0] es el nombre del programa
    // --> lectores_escritores 5 3   (5 lectores y 3 escritores)
    int num_lectores=atoi(argv[1]); 
    int num_escritores=atoi(argv[2]);
    



    pthread_t hilos_lectores[num_lectores]; // hilos que seran lectores
    pthread_t hilos_escritores[num_escritores]; // hilos que seran escritores



    int ids_lectores[num_lectores],ids_escritores[num_escritores];

    pthread_mutex_init(&mutex,NULL); // se inicializa el mutex
    sem_init(&semaforo,0,1); // se inicialiaz el semaforo con 0 porque es para hilos y con contador en 1


    //Creacion de hilos
    // Lectores
    for(int i=0;i<num_lectores;i++) 
    {
        ids_lectores[i]=i+1;
        pthread_create(&hilos_lectores[i],NULL,lector,(void *)&ids_lectores[i]); // argumentos: hilo, NULL, funcion a hacer, id del hilo lector
    }
    // Escritores
    for(int i=0;i<num_escritores;i++) 
    {
        ids_escritores[i]=i+1;
        pthread_create(&hilos_escritores[i],NULL,escritor,(void *)&ids_escritores[i]); // argumentos: hilo, NULL, funcion a hacer, id del hilo escritor
    }



    // Esperar a que los hilos terminen
    // Lectores
    for(int i=0;i<num_lectores;i++) 
    {
        pthread_join(hilos_lectores[i],NULL); // argumentos: hilo, NULL
    }
    // Escritores
    for(int i=0;i<num_escritores;i++) 
    {
        pthread_join(hilos_escritores[i],NULL); // argumentos: hilo, NULL
    }

    pthread_mutex_destroy(&mutex); // destruir el mutex
    sem_destroy(&semaforo); // destruir el semaforo

    return 0;
}
