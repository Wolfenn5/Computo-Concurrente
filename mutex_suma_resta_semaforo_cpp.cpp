#include <iostream>
#include <thread>
#include <mutex>
#include <stdlib.h>
// #include <semaphore.h> // biblioteca para semaforos pero de c

#include <semaphore> // biblioteca de semaforos pero de c++ version 20

//asi que se compila             --> g++ nombre.cpp -o nombre -std=c++20
// o por ejemplo en mac          --> clang++ -std=c++11 -pthread



int balance=0;

// sem_t semaforo; 

// el valor que va dentro de los simbolos < >  indica la cantidad de espacios disponibles
// el valor que va dentro de ( ) indica la cantidad de permisos
std::counting_semaphore<1> semaforo(1); //inicializar el contador del semaforo en 1


// este al ser binario no se ponen los espacios, solo el contador
//std::binary_semaphore semaforo(1); // inicializar un contador binario (un tipo de mutex)



void suma (int iteraciones)
{
    for (int i=0; i<iteraciones; i++)
    {
        //sem_wait(&semaforo); // bloquear con semaforo C
        semaforo.acquire(); // bloquear con semaforo C++
        balance= balance+1;
        semaforo.release(); // desblloquear con semaforo C++
        //sem_post(&semaforo); // desbloquear con semaforo C
    }  
}



void resta (int iteraciones)
{
    for (int i=0; i<iteraciones; i++)
    {
        //sem_wait(&semaforo); // bloquear con semaforo C
        semaforo.acquire(); // bloquear con semaforo C++
        balance= balance-1;
        semaforo.release(); // desblloquear con semaforo C++
        //sem_post(&semaforo); // desbloquear con semaforo C
    }  
}


int main(int argc, char const *argv[])
{
    int iteraciones= std::atoi(argv[1]);

    std::thread hilo1(suma, iteraciones);
    std::thread hilo2(resta, iteraciones);


    //sem_init(&semaforo,0,1); // semaforo de hilos con valor de 1 en el contador

    hilo1.join();
    hilo2.join();


    // destruir semaforo para liberar recursos
    //sem_destroy(&semaforo);

    std::cout<<"\nEl balance es "<<balance<<"\n";
    return 0;
}
