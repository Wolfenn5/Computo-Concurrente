#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <semaphore.h>

/* Este programa simula el trabajo de un policia y varios peatones*/


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // mutex
pthread_cond_t condicion = PTHREAD_COND_INITIALIZER; // variable de condicion para la señal

int luz_semaforo=0; //0 sera el rojo y 1 sera verde


sem_t semaforo_peaton; // semaforo para indicarle al peaton cuando cruza




void * semaforo(void * arg)
{
	sleep(3); // simular el tiempo en el que el semaforo esta en rojo
	pthread_mutex_lock(&mutex); // el policia recibe la peticion del peaton para poder liberar de forma temporal, luego el policia bloquea el semaforo para poder cambiar la luz del semaforo

	luz_semaforo=1; // el policia cambia l luz a verde y avisa al peaton y avisa al peaton
	printf("\nHe cambiado el color de la luz del semaforo, puedes cruzar\n");
	pthread_cond_signal(&condicion); // se manda una señal para avisarle al peaton 
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}





void * peaton(void * arg)
{
	int * id = (int *) arg;
	printf("\nSoy el peaton %d y estoy esperando a poder cruzar\n",*id);
	pthread_mutex_lock(&mutex);
	while(luz_semaforo == 0) //mientras la luz del semaforo este en rojo
    {
		pthread_cond_wait(&condicion,&mutex);//el peaton va a estar volteando y esperara a poder cruzar... (cada que voltea a ver al policia, le desbloquea de forma temporal a la condicion)
    }

    /* Seccion del ejercicio de repaso para que los peatones pasen uno por uno*/
	// en si, se añadio un semaforo para bloquear a los peatones uno por uno
    sem_wait(&semaforo_peaton); // el semaforo indica que el peaton puede curzr y se bloquea para que los demas peatones (hilos) no crucen
	printf("\nGracias por dejarme cruzar (peaton %d)\n",*id);
    sleep(3); // simular el tiempo que tarda en cruzar cada peaton
    pthread_cond_signal(&condicion); // cada peaton despues de cruzar, "despertara" mandara una señal a otro peaton que este esperando en pthread_cond_wait, asegurando que todos puedan cruzar
	pthread_mutex_unlock(&mutex);
    sem_post(&semaforo_peaton); // se desbloquea el semaforo cuando el peaton ya cruzo, indicando a los demas peatones que pueden cruzar

	pthread_exit(NULL);
}





int main(int argc, char * argv[])
{
    // Parametros del main sera el numero de peatones
    // --> programa 5
	int num_peatones=atoi(argv[1]);


	// Declarar los hilos
	pthread_t hilo_policia;
    pthread_t hilos_peaton[num_peatones];


    // Crear los hilos
	pthread_create(&hilo_policia, NULL, semaforo, NULL); // policia

    // Peatones
	for(int i=0; i<num_peatones; i++)
    {
		int * id= malloc(sizeof(int)); // cada peaton tiene su identificador "unico"
		* id= i+1;
		pthread_create(&hilos_peaton[i], NULL, peaton, (void *)id);
	}


	// Esperar a que los hilos terminen
    // Policia
	pthread_join(hilo_policia, NULL);

    // Peatones
	for(int i=0;i<num_peatones;++i)
		pthread_join(hilos_peaton[i],NULL);


	pthread_mutex_destroy(&mutex); // destruir el mutex
	pthread_cond_destroy(&condicion); // destruir la variable de condicion

    return 0;
}