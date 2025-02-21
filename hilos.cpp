#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <chrono> // biblioteca para trabajar con el tiempo pero de forma nativa
// pthread es el estandar de POSIX y trabaja mejor en UNIX pero thread es mas escalable y portable

// program counter es global
int pc=0;


void ejecutaInstruccion(std::string instruccion)
{
    std::cout<<"PC: "<<pc<<" --> Ejecutando: "<<instruccion<<"\n";
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); //dormir por 500 milisegundos el hilo de ejecucion
}



int main(int argc, char const *argv[])
{
    std::vector<std::string> instrucciones=
    {
        "Cargar datos del proceso A",
        "Cargar datos al proceso B",
        "Sumar datos del proceso A",
        "Multiplicar datos del proceso B",
        "Sumar los resultados del proceso A con los datos del proceso B"
    };



    // recorrer cada instruccion y aumentar el pc
    // for (int i=0; i<instrucciones.size(); i++)
    for(auto &instruccion: instrucciones) // auto es un "tipo de dato" para trabajar con cualquier elemento de forma automatica
    {
        ejecutaInstruccion(instruccion);
    }
    std::cout<<"PC al terminar la ejecucion --> "<<pc<<"\n";
    return 0;
}
