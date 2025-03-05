#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define BUFFER_SIZE 10
#define N 10

int buffer[BUFFER_SIZE];
int count=0; // variable para controlar la señal de espera

pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t no_lleno= PTHREAD_COND_INITIALIZER;
pthread_cond_t no_vacio= PTHREAD_COND_INITIALIZER;



void * productor (void * arg)
{
    for (int i=0; i<N; i++)
    {
        pthread_mutex_lock(&mutex);
        while (count == BUFFER_SIZE)
        {
            // esperar
            pthread_cond_wait(&no_lleno, &mutex);
        }
        buffer[count++]= i;
        printf("\nEl productor produjo: %d",i);
        // una vez que se pueda producir, necesitamos despertar a otro hilo
        pthread_cond_signal(&no_vacio); // despertar al consumidor
        pthread_mutex_unlock(&mutex);
        sleep(1); // simular el tiempo de produccion
    }
    pthread_exit(NULL);
}


void * consumidor (void * arg)
{
    for (int i=0; i<N; i++)
    {
        pthread_mutex_lock(&mutex);
        while (count == 0) // esperar si el buffer esta vacio
        {
            // esperar
            pthread_cond_wait(&no_vacio, &mutex);
        }
        int item= buffer[--count];
        printf("\nConsumidor consumio: %d",item);
        pthread_cond_signal(&no_lleno); // cuando el buffer esta vacio o tiene un espacio disponible, se despierta al hilo productor
        pthread_mutex_unlock(&mutex);
        //sleep(2);
        usleep(rand()% 1500000);
    }
    pthread_exit(NULL);
}



int main(int argc, char const *argv[])
{
    srand(time(NULL));
    pthread_t hiloProductor, hiloConsumidor;

    pthread_create(&hiloProductor, NULL, productor, NULL);
    pthread_create(&hiloConsumidor, NULL, consumidor, NULL);

    pthread_join(hiloProductor,NULL);
    pthread_join(hiloConsumidor,NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&no_lleno);
    pthread_cond_destroy(&no_vacio);
    return 0;
}
