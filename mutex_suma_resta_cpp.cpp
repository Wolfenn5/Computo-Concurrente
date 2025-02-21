#include <iostream>
#include <thread>
#include <mutex>
#include <stdlib.h>


std::mutex mutex;
int balance=0;


void suma (int iteraciones)
{
    for (int i=0; i<iteraciones; i++)
    {
        mutex.lock();
        balance= balance+1;
        mutex.unlock();
    }  
}



void resta (int iteraciones)
{
    for (int i=0; i<iteraciones; i++)
    {
        mutex.lock();
        balance= balance-1;
        mutex.unlock();
    }  
}


int main(int argc, char const *argv[])
{
    int iteraciones= std::atoi(argv[1]);

    std::thread hilo1(suma, iteraciones);
    std::thread hilo2(resta, iteraciones);


    hilo1.join();
    hilo2.join();



    std::cout<<"\nEl balance es "<<balance<<"\n";
    return 0;
}
