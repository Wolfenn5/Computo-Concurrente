#include <iostream>
#include <semaphore>
#include <chrono>
#include <mutex>
#include <vector>
#include <thread>

#include <cstdlib>
#include <ctime>


/* Este programa sirve para evitar deadlock (tiempos muertos en ejecucion), hay que evitar que los hilos queden bloqueados entre si*/
// como esta counting_semaphore al compilar se usa: 
//--> g++ nombre.cpp -o nombre -std=c++20



std::mutex mutex_libro; // mutex que sera considerado el libro
std::mutex mutex_lectores; // mutex para los lectores
std::counting_semaphore<1> semaforo_escritor(1); // semaforo para bloquear el acceso de los escritores
// puede ser tanto counting_semaphore o binary_semaphore


int contador_lectores=0; // "bandera" para indicar (si hay x lectores hacer x cosa)




void lector (int id_hilo)
{
    mutex_lectores.lock();
    contador_lectores++; // indicar que un lector esta leyendo
    if (contador_lectores == 1) // si hay un lector, se bloquea a los escritores por la condicion de que se puede leer pero no escribir
    {
        semaforo_escritor.acquire(); // bloquear, si hay un lector y no es el ultimo se puede seguir leyendo
    }
    mutex_lectores.unlock(); // esta parte es la seccion critica (donde hay un acceso a memoria y puede ser modificada)


    std::cout<<"El lector "<<id_hilo<<" esta ocupando el libro\n";
    std::this_thread::sleep_for(std::chrono::seconds(2)); // dormir al hilo 2 segundos
    std::cout<<"\nYa termine de leer\n";

    mutex_lectores.lock(); // bloquear la variable
    contador_lectores--; // indicar que un lector acabo de leer
    if (contador_lectores == 0)
    {
        semaforo_escritor.release(); // se libera el acceso al escritor siempre y cuando no haya lectores leyendo el libro
    }
    mutex_lectores.unlock(); // para evitar el deadlock se desbloquea el mutex al final
}



void escritor (int id_hilo)
{
    semaforo_escritor.acquire(); // activar el semaforo para no dejar pasar a mas escritores
    mutex_lectores.lock(); // bloquear para que los lectores no lean mientras se escribe en el libro
    std::cout<<"\n El escritor "<<id_hilo<<" esta escribiendo...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2)); // dormir al hilo 2 segundos
    std::cout<<"\nYa termine de escribir\n";
    semaforo_escritor.release(); // liberar el semaforo para dejar pasar a los demas escritores
    mutex_lectores.unlock(); // se desbloquea para que los lectores ya puedan leer
}




int main(int argc, char const *argv[])
{
    int lectores, escritores;

    
    // Parametros del main
    // --> lectores_escritores_clase 5 3   (5 lectores y 3 escritores)







    lectores=std::atoi(argv[1]);
    escritores=std::atoi(argv[2]);


    /*---------------------------------------------------------------------------------------------------------------------------------*/
    /* Se usa esto para evitar que al ejecutar, siempre sean primero lectores y luego escritores (depende de como esten declarados en el for, pueden ser tambien primero escritores y luego lectores)*/
    // pero hay que comentar las lineas de arriba (lo de argumentos de main lectores, escritores nadamas) y lo que esta debajo de las lineas




    // int n_hilos= atoi(argv[1]);
    // srand(time(NULL));
    // std::vector<std::thread> hilos;
    // // rand_max es el valor maximo que puede ocuparse para generar como aleatorio
    
    // for (int i=0; i<n_hilos; i++)
    // {
    //     double random= (double)rand()/RAND_MAX; // generar numeros aleatorios entre 0 y 1
    //     if (random <= 0.7)
    //     {
    //         hilos.push_back(std::thread(lector,i));
    //     }
    //     else
    //     {
    //         hilos.push_back(std::thread(escritor,i));
    //     }
        
    // }
    
    // for (auto & hilo : hilos)
    // {
    //     hilo.join();
    // }

    
    /*---------------------------------------------------------------------------------------------------------------------------------*/




    // Declarar hilos lectores y escritores
    std::vector<std::thread> hilos_lectores;
    std::vector<std::thread> hilos_escritores;
    // Otra forma es
    // std::vector<std::thread> hilos;
    // for (int i=0; i<n/2; i++)
    // {
    //     hilos.push_back(std::thread(escritores, parametros));
    // }
    


    // Inicializar hilos
    //Lectores
    for (int i=0; i<lectores; i++)
    {
        hilos_lectores.push_back(std::thread(lector,i));
    }
    // Escritores
    for (int i=0; i<lectores; i++)
    {
        hilos_lectores.push_back(std::thread(escritor,i));
    }
    

    // Esperar a que los hilos terminen
    // Lectores
    for (auto & hilo : hilos_lectores)
    {
        hilo.join();
    }
    // Escritores
    for (auto & hilo : hilos_escritores)
    {
        hilo.join();
    }
    


    return 0;
}
