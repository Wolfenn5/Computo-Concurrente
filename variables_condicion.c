#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

// Inicializar un mutex cuando es variable global, cuando el mutex y la variable de condicion son globales
pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER; // mutex
pthread_cond_t condicion= PTHREAD_COND_INITIALIZER; // variable

int senial_simulada=0; // recurso compartido




void * hilo_secundario(void *argc)
{
    sleep(3);
    pthread_mutex_lock(&mutex); // como cond_wait libera temporalmente el mutex, nos aseguramos de que nosotros (quienes vamos a modificar la señal), tengamos bloqueado el acceso a otros hilos
    senial_simulada=1;
    printf("\nTe mandare la señal\n");
    pthread_cond_signal(&condicion);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}


int main(int argc, char const *argv[])
{
    pthread_t hilo;

    // Crear el hilo secundario
    pthread_create(&hilo, NULL, hilo_secundario, NULL);

    printf("\nMande a ejecutar el hilo secundario y esperare la señal\n");
    



    pthread_mutex_lock(&mutex); // Siempre se va a bloquear el mutex para evitar que un hilo ajeno, acceda al recurso compartido
    /* Seccion critica del programa*/
    while (senial_simulada == 0) // La condicion que activa pthread_cond
    {
        pthread_cond_wait(&condicion, &mutex); // Liberar temporalmente el mutex, dependiendo de la señal. Se desbloquea mientras se espera la señal
    }  
    printf("\nRecibi la señal y continuo con el programa\n");
    pthread_mutex_unlock(&mutex);


    
    pthread_join(hilo,NULL); // esperar al hilo
    pthread_mutex_destroy(&mutex); // destruir el mutex   
    pthread_cond_destroy(&condicion); // destruir pthread_cond

    return 0;
}
