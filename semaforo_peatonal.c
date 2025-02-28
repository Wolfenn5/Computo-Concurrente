#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

/* Este programa simula el trabajo de un policia y un peaton*/


pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER; // mutex
pthread_cond_t condicion= PTHREAD_COND_INITIALIZER; // variable


int luz_semaforo=0;// 0 sera el rojo y 1 sera verde



void * semaforo(void * argc)
{
    sleep(3); // simular el tiempo en el que el semaforo esta en rojo
    pthread_mutex_lock(&mutex); // el policia recibe la peticion del peaton para poder liberar de forma temporal, luego el policia bloquea el semaforo para poder cambiar la luz del semaforo

    luz_semaforo=1; // el policia cambia l luz a verde y avisa al peaton
    printf("\nHe cambiado la luz del semaforo\n");
    pthread_cond_signal(&condicion);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}



void * peaton(void * argc)
{

    int * id= (int*) argc;




    printf("\nSoy el peaton y estoy esperando poder cruzar", *id);
    pthread_mutex_lock(&mutex);

    while (luz_semaforo == 0) // mientras la luz del semaforo este en rojo
    // Si hubiera un if en vez de un while, el peaton solo voltearia una vez o alomejor el policia le diria que esta en verde y el peaton no oye
    {
        pthread_cond_wait(&condicion, &mutex); // el peaton estara volteando y esperara para poder cruzar (cada que voltea a ver al policia, le desbloquea de forma temporal la condicion)
    }


    printf("\nGracias por dejarme cruzar\n");
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
    
}







int main(int argc, char const *argv[])
{

    /* Version con varios hilos*/
    int num_peatones= atoi(argv[1]);
    pthread_t hilo_policia, hilos_peaton[num_peatones];
    pthread_create(&hilo_policia,NULL, semaforo, NULL);
    int * id;
    for (int i=0; i<num_peatones; i++)
    {
        id= id+1;
        pthread_create(&hilos_peaton[i],NULL, peaton, (void *)id);
    }


    
    pthread_join(hilo_policia,NULL);
    for (int i=0; i<num_peatones; i++)
    {
        pthread_join(hilos_peaton[i],NULL);
    }
    
    /*----------------------------------------------------------------------*/






    // // Declarar los hilos
    // pthread_t hilo_policia;
    // pthread_t hilo_peaton;


    // // Crear los hilos
    // pthread_create(&hilo_policia, NULL, semaforo, NULL);
    // pthread_create(&hilo_policia, NULL, peaton, NULL);



    // // Esperar a que los hilos terminen
    // pthread_join(hilo_policia,NULL);
    // pthread_join(hilo_peaton, NULL);
    

    // Liberacion
    pthread_mutex_destroy(&mutex); // destruir el mutex
    pthread_cond_destroy(&condicion); // destruir variable de condicion


    return 0;
}
