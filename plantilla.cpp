// Otra forma de compilar      --> g++ nombre.cpp -o nombre -std=c++20
// Compilar en mac             --> clang++ -std=c++11 -lpthread nombre.cpp -o nombre  solo clang si es en c

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



/* --------- Variables de Condicion --------- */
// Declaracion de variable de condicion
pthread_cond_t variable_condicion= PTHREAD_COND_INITIALIZER;
// Notificaciones 
pthread_cond_signal(&variable_condicion); // despierta a un solo hilo
pthread_cond_broadcast(&variable_condicion); // despierta a un solo hilo
// Destruccion de variable de condicion
pthread_cond_destroy(&variable_condicion);
// Esperar por una señal
pthread_cond_wait(&variable_condicion, &mutex); // Liberar temporalmente el mutex, dependiendo de la señal. Se desbloquea mientras se espera la señal


/* -------------- C++ --------------*/
#include <iostream>
#include <thread> // pthread es el estandar de POSIX y trabaja mejor en UNIX pero thread es mas escalable y portable
#include <chrono> // biblioteca para trabajar con el tiempo pero de forma nativa de C++
#include <mutex> 
#include <semaphore> // semaforos
#include <condition_variable> // biblioteca para variables de condicion


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
std::counting_semaphore<1> semaforo(1); // inicializar el contador del semaforo en 1 y el valor maximo de ese contador sera 1 <1>
// Bloquear con semaforo
semaforo.acquire();
// Desbloquear con semaforo
semaforo.release();
// Destruccion de semaforos se hace de forma automatica



/* --------- Mutex --------- */
// Declaracion de mutex
std::mutex mutex;
// Sirve para liberar de forma "semi-automatica" el mutex en vez de usar lock y unlock
std::lock_guard<std::mutex> lock (mutex); // sirve para liberar de forma "semi-automatica" el mutex en vez de usar lock y unlock  (el bloque de codigo que este dentro de {} )
// Bloquear con mutex
mutex.lock();
// Desbloquear con mutex
mutex.unlock();
// Destruccion de mutex se hace de forma automatica



/* --------- Variables de Condicion --------- */
// Declaracion de variable de condicion
std::condition_variable variable_condicion;
// Notificaciones 
variable_condicion.notify_one(); // notificar a otro hilo que se modifico algo ; es equivalente a pthread_cond_signal()     que despierta un hilo
variable_condicion.notify_all(); // notificar a otro hilo que se modifico algo ; es equivalente a pthread_cond_broadcast()  que despierta a todos los hilos