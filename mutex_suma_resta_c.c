#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>


int balance;

pthread_mutex_t mutex; // crear el mutex

void * suma (void * arg)
{
	int * iteraciones= (int *)arg; 
	for (int i=0; i<*iteraciones; i++)
	{
		pthread_mutex_lock(&mutex); // bloquear
		balance= balance+1;
		pthread_mutex_unlock(&mutex); // desbloquear
	}
	return NULL;
}



void * resta (void * arg)
{
	int * iteraciones= (int *)arg; 
	for (int i=0; i<*iteraciones; i++)
	{

		pthread_mutex_lock(&mutex); // bloquear
		balance= balance-1;
		pthread_mutex_unlock(&mutex); // desbloquear
	}
	return NULL;
}


int main(int argc, char const *argv[])
{
	int iteraciones= atoi(argv[1]);
	pthread_t sumador, restador;  //crear 2 hilos que sumen y resten


	pthread_create(&sumador, NULL, suma, (void*)&iteraciones);
	pthread_create(&restador, NULL, resta, (void*)&iteraciones);


	// esperar a que terminen los hilos
	pthread_join(sumador, NULL);
	pthread_join(restador, NULL);

	printf("La variable balanceo al final es:%d\n",balance);
	return 0;
}
