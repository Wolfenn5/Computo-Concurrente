// Otra forma de compilar      --> g++ nombre.cpp -o nombre -std=c++20
// Compilar en mac             --> clang++ -std=c++11 -pthread

/* -------------- C -------------- */
#include <stdio.h>
#include <semaphore.h> // semaforos
#include <pthread.h> // hilos
#include <unistd.h> // para dormir hilos o manejarlos con fork, exec, getpid 


/* --------- Hilos --------- */
// Declaracion de hilos
pthread_t hilo1, hilo2;
// Creacion de hilos
pthread_create(&hilo1, NULL, funcion_a_hacer, (void*)&iteraciones); // ver programa mutex_suma_resta_c
// Esperar a que un hilo termine
pthread_join(hilo1, NULL);
// Forzar a que un hilo termine
pthread_exit(hilo1, NULL);



/* --------- Semaforos --------- */
// Declaracion de semaforos
sem_t semaforo;
// Inicializar el semaforo 
sem_init(&semaforo,0,1); // los parametros son la direccion de memoria del semaforo, tipo de semaforo (0 para hilos) y (1 para procesos), valor del contador
// Bloquear con semaforo
sem_wait(&semaforo); 
// Desbloquear con semaforo}
sem_post(&semaforo);
// Destruccion de semaforos
sem_destroy(&semaforo);



/* --------- Mutex --------- */
// Declaracion de mutex
pthread_mutex_t mutex;
// Bloquear con mutex
pthread_mutex_lock(&mutex);
// Desbloquear con mutex
pthread_mutex_unlock(&mutex);
// Destruccion de mutex
pthread_mutex_destroy(&mutex);




/* -------------- C++ --------------*/
#include <iostream>
#include <thread> // pthread es el estandar de POSIX y trabaja mejor en UNIX pero thread es mas escalable y portable
#include <chrono> // biblioteca para trabajar con el tiempo pero de forma nativa de C++
#include <mutex> 
#include <semaphore> // semaforos


/* --------- Hilos --------- */
// Declaracion de hilos
std::thread hilo1(suma, iteraciones); // funcion suma y parametro de iteraciones a hacer ver programa  mutex_suma_resta_semaforo_cpp
// Creacion de hilos
std::thread hilo1(funcion_a_hacer, std::ref(iteraciones); // si iteraciones no es variable global
// Dormir hilos
std::this_thread::sleep_for(std::chrono::milliseconds(500)); // dormir por 500 milisegundos el hilo de ejecucion
// Esperar a que un hilo termine
hilo1.join();



/* --------- Semaforos --------- */
// Declaracion de semaforos
std::counting_semaphore<1> semaforo(1); // inicializar el contador del semaforo en 1
// Bloquear con semaforo
semaforo.acquire();
// Desbloquear con semaforo
semaforo.release();
// Destruccion de semaforos se hace de forma automatica



/* --------- Mutex --------- */
// Declaracion de mutex
std::mutex mutex;
// Bloquear con mutex
mutex.lock();
// Desbloquear con mutex
mutex.unlock();
// Destruccion de mutex se hace de forma automatica