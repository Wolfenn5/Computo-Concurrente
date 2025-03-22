// Otra forma de compilar      --> g++ nombre.cpp -o nombre -std=c++20
// Compilar en mac             --> clang++ -std=c++11 -lpthread nombre.cpp -o nombre  solo clang si es en c


/* -------------- CUDA -------------- */
// Calcular el numero de hilos a ocupar
// Calcular el numero de bloques necesario para cubrir 1 millon de elementos con un tamaño de bloque de 1024 hilos
// int N= 1000000000;
// int tambloque= 1024;
// int numbloques= (N+tambloque-1) / tambloque;
// La operacion seria (1,000,000,000 + 1024 -1) / 1024 = 976,567 numbloques

// Calcular el tamaño de la malla "tamanoMalla" utilizando el numero de bloques anterior de 976,567
// La operacion seria (1,000,000,000 + 976,567 -1) / 976,567 = 1024.99 = tamanoMalla= (1025,1025)

// Calcular los hilos totalHilos multiplicando el tamaño de la malla en cada dimension por el tamaño del bloque
// totalHilos= 1025* 1025 * 1024= 1,075,840,000;




/* Ejemplo sencillo de dim3 con gridsize y blocksize*/

// #include <stdio.h>
// #include <cuda_runtime.h>


// __global__ void kernel()
// {
// int threadId = blockIdx.x * blockDim.x + threadIdx.x;
// printf("Hilo ID: %d\n", threadId);
// }



// int main()
// {
// dim3 gridSize(2, 1, 1); //malla de 2 bloques en x, 1 bloque en y, 1 bloque en z
// // el numero total de bloques en la cuadricula es 2x1x1= 2

// dim3 blockSize(3, 1, 1); //bloques de 3 hilos en x, 1 hilo en y, 1 hilo en z
// // cada bloque tendra 3x1x1= 3 hilos


// // Asi el numero total de hilos en toda la cuadricula va a ser de 2x3= 6
// kernel<<<gridSize, blockSize>>>();
// cudaDeviceSynchronize();
// return 0;
// }



// Obtener datos del dispositivo GPU
cudaDeviceProp propiedades;
cudaGetDeviceProperties(&propiedades,0);

// Calcular el numero de hilos a ocupar
// Sacar propiedades del dispositivo
cudaDeviceProp propiedades;
cudaGetDeviceProperties (&propiedades,0);
int tam_bloque= propiedades.maxThreadsPerBlock; // calcular tamaño optimo del bloque
int num_bloques= (N+tam_bloque-1) / tam_bloque; // calcular el numero de bloques ; N es el numero de datos (en este caso el tamaño de la matriz), la formula es universal
// Si se quiere saber el numero de hilos maximo se multiplica tam_bloque*numero_bloques
// Definir el tamaño de la malla y el bloque
dim3 tamanio_bloque(num_bloques,num_bloques);
dim3 tamanio_malla((N+num_bloques-1) / num_bloques,(N+num_bloques-1) / num_bloques);

// Obtener el id del hilo
int id_hilo= blockIdx.x * blockDim.x + threadIdx.x; // bloque 0 dimension bloque 512 + id hilo 0 seria el hilo 512
// Declaracion en el host (CPU)
float *A_dispositivo= (float*) malloc(sizeof(float)*dimension*dimension2);
// Declaracion en el dispositivo (GPU)
cudaMalloc(&A_dispositivo, dimension*sizeof(int));
// Mover la memoria del host al dispositivo
cudaMemcpy(A_dispositivo, A_host, dimension*sizeof(int),cudaMemcpyHostToDevice);
// Regresar la informacion del dispositivo al host
cudaMemcpy(resultado_host, resultado_dispositivo,sizeof(float)*n,cudaMemcpyDeviceToHost);
// Esperar a los hilos del dispositivo (GPU)
cudaDeviceSynchronize(); // es el equivalente a .join en hilos. Es importante utilizarlo para no trabar la GPU
// Liberar recursos del host (CPU)
free (matriz_host);
// Liberar recursos del dispositivo (GPU)
cudaFree(matriz_dispositivo);




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