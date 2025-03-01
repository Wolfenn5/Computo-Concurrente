//este programa simula el trabajo de un policia y un peaton que quiere cruzar
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condicion = PTHREAD_COND_INITIALIZER;

int luz_semaforo = 0;//0 para rojo y 1 para verde

void * semaforo(void * arg){
	sleep(1);//simulamos el tiempo en que el semaforo esta en rojo
	pthread_mutex_lock(&mutex);//el policia recibe la peticion del peaton para poder liberar de forma temporal... entonces, el policia bloquea el semaforo para solamente el poder cambiar el color de la luz
	luz_semaforo=1;//el policia cambia la luz a color verde
	//y le avisa al peaton que ya la cambió
	printf("\nHe cambiado el color de la luz del semaforo, puedes cruzar\n");
	pthread_cond_signal(&condicion);
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}

void * peaton(void * arg){
	int * id = (int *) arg;
	printf("\nSoy el peaton %d y estoy esperando a poder cruzar\n",*id);
	pthread_mutex_lock(&mutex);
	while(luz_semaforo==0)//mientras la luz del semaforo este en rojo
		pthread_cond_wait(&condicion,&mutex);//el peaton va a estar volteando y esperara a poder cruzar... (cada que voltea a ver al policia, le desbloquea de forma temporal a la condicion)
	printf("\nGracias por dejarme cruzar (peaton %d)\n",*id);
	pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&condicion); // cada peaton despues de cruzar, despertara a otro peaton que este esperando en pthread_cond_wait, asegurando que todos puedan cruzar
	pthread_exit(NULL);
}

int main(int argc, char * argv[]){
	int num_peatones=atoi(argv[1]);
	//declaramos nuestros hilos
	pthread_t hilo_policia, hilos_peaton[num_peatones];
	pthread_create(&hilo_policia,NULL,semaforo,NULL);
	for(int i=0;i<num_peatones;++i){
		int * id = malloc(sizeof(int));//cada ranita tiene su identificador "unico"
		* id = i+1;
		pthread_create(&hilos_peaton[i],NULL,peaton,(void *)id);
	}
	//esperamos a que los hilos terminen
	pthread_join(hilo_policia, NULL);
	for(int i=0;i<num_peatones;++i)
		pthread_join(hilos_peaton[i],NULL);
	//liberacion del mutex y la variable de condicion
	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&condicion);
}