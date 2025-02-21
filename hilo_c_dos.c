#include <stdio.h>
#include <pthread.h>
#include <unistd.h> // psra dormir hilos o manejarlos con fork, exec, getpid 

// -lpthread bandera que a veces es necesaria al momento de compilar





void * muestra_Contador(void * arg) // estas funciones necesitan recibir los parametros como un apuntador de tipo void, si la funcion la va a usar un hilo, debe tener solo un apuntador
{
    int * contador= (int*)arg;
    while (*contador<=10)
    {
        printf("El contador es: %d\n", *contador);
        sleep(1); // dormir cada 2 segundos
    }
    
    
    //pthread_exit(); // para que el hilo acabe lo que tenga que hacer
    return NULL; // regresar un apuntador nulo porque la funcion esta como void *
}


int main(int argc, char const *argv[])
{
    int * contador=0;
    pthread_t hilo1; // declarar el hilo1 como un hilo de tipo pthread   ;    se asignan espacios de memoria para la pila (variables del hilo), registros de cpu y contador de programa
    pthread_create(&hilo1, NULL, muestra_Contador, (void*) &contador); // aqui se le asigna que instrucciones hara el hilo    


    for (int i=0; i<10; ++i)
    {
        printf("\nEstoy incrementando el contador: ");
        contador= contador + 1;
        sleep(1);
    }
    

    pthread_join(hilo1, NULL);
    printf("Termine de actualizar\n");

    return 0;
}
