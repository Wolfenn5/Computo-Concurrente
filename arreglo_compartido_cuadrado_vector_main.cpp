#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib> // para usar parametros del main
#include <vector>




void calculaCuadrados(std::vector<int> &arreglo, int inicio, int fin, int id)
{
    for (int i=inicio; i<fin; i++)
    {
        arreglo[i]= arreglo[i]*arreglo[i];
        std::this_thread::sleep_for(std::chrono::seconds(1)); //dormir el hilo por un segundo
        std::cout<<"Yo soy el hilo "<<id<<" El valor que modifique es: "<<arreglo[i]<<"\n";
    }
}


//los parametros del main se leen desde argv [] porque argv[] es el nombre del programa
int main(int argc, char const *argv[])
{
    int TAMANIO= std::atoi(argv[1]); //tamaño del arreglo con argumento del main
    std::vector <int> (arreglo);

    // inicializar el arreglo
    for (int i=0; i<TAMANIO; i++)
    {
        arreglo[i]=i;
    }


    std::cout<<"\nEl arreglo original es:\n";
    for (int i=0; i<TAMANIO; i++)
    {
        std::cout<<arreglo[i]<<" ";
    }
    std::cout<<"\n\n";

    int mitad= TAMANIO/2;


    // crear hilos
    std::thread hilo1(calculaCuadrados, std::ref(arreglo),0, mitad, 1);
    std::thread hilo2(calculaCuadrados, std::ref(arreglo), mitad, TAMANIO, 2);


    // esperar a que los hilos terminen
    hilo1.join();
    hilo2.join();


    std::cout<<"\nEl arreglo modificado es:\n";
    for (int i=0; i<TAMANIO; i++)
    {
        std::cout<<arreglo[i]<<" ";
    }
    
    return 0;
}
