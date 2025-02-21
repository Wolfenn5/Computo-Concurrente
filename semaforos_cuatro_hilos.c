#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <semaphore.h> // biblioteca para semaforos


int balance=0;



sem_t semaforo; // declarar el semaforo

pthread_mutex_t mutex; // crear el mutex




void * suma (void * arg)
{
	int * iteraciones= (int *)arg; 
	for (int i=0; i<*iteraciones; i++)
	{
		sem_wait(&semaforo); // bloquear con semaforo
        // el parametro es la direccion de memoria y en automatico decrementa el valor del contador del propio semaforo
		balance= balance+1;
		sem_post(&semaforo); // desbloquear con semaforo
        // el parametro es la direccion de memoria del semaforo y en automatico incrementa el valor del contador del propio semaforo   
	}
	return NULL;
}



void * resta (void * arg)
{
	int * iteraciones= (int *)arg; 
	for (int i=0; i<*iteraciones; i++)
	{
		sem_wait(&semaforo); // bloquear con semaforo
        // el parametro es la direccion de memoria y en automatico decrementa el valor del contador del propio semaforo
		balance= balance-1;
		sem_post(&semaforo); // desbloquear con semaforo
        // el parametro es la direccion de memoria del semaforo y en automatico incrementa el valor del contador del propio semaforo   
	}
	return NULL;
}




int main(int argc, char const *argv[])
{
	int iteraciones= atoi(argv[1]);
	pthread_t sumador1, sumador2, restador1, restador2;  //crear 2 hilos que sumen y resten


    /* Se pone el semaforo con valor de 1 porque se pide que sea concurrente, de esa forma asi entrara un solo hilo y al terminar, llegara otro;  de otra forma seria paralelo*/
    sem_init(&semaforo,0,1); //inicializar el semaforo c
    // los parametros son la direccion de memoria del semaforo, tipo de semaforo (0 para hilos) y (1 para procesos), valor del contador




	pthread_create(&sumador1, NULL, suma, (void*)&iteraciones);
	pthread_create(&restador1, NULL, resta, (void*)&iteraciones);
    pthread_create(&sumador2, NULL, suma, (void*)&iteraciones);
	pthread_create(&restador2, NULL, resta, (void*)&iteraciones);


	// esperar a que terminen los hilos
	pthread_join(sumador1, NULL);
	pthread_join(restador1, NULL);
    pthread_join(sumador2, NULL);
	pthread_join(restador2, NULL);


    // destruir los semaforo para liberar los recursos que ocupa
    sem_destroy(&semaforo);

    

	printf("La variable balanceo al final es:%d\n",balance);
	return 0;
}
