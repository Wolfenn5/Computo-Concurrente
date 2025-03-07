#include <iostream>
#include <thread>
#include <barrier>
#include <chrono>
#include <vector>

#define NUMCABALLOS 4
#define ETAPAS 3

std::barrier barrera(NUMCABALLOS); //a diferencia de c, se pasa el valor de contador a la barrera desde la declaracion




void carrera (int id)
{
    for (int i=0; i<ETAPAS; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000));
        std::cout<<"\nEl caballo "<<id<<" ha terminado la etapa "<<i<<" y va a esperar";
        // esperar a que todos los caballos lleguen
        barrera.arrive_and_wait(); 
    }
    std::cout<<"\nEl caballo "<<id<<" ha terminado la carrera\n";
}


int main(int argc, char const *argv[])
{
    // Vector de hilos
    std::vector<std::thread> caballos;
    for (int i=0; i<NUMCABALLOS; i++)
    {
        caballos.push_back(std::thread(carrera,i));
    }
    

    // Esperar a que los caballos terminen
    for(auto & caballo : caballos)
    {
        caballo.join();
    }

    return 0;
}
