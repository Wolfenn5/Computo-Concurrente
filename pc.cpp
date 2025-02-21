#include <iostream>
#include <thread>
#include <chrono> // biblioteca para trabajar con el tiempo pero de forma nativa
// pthread es el estandar de POSIX y trabaja mejor en UNIX pero thread es mas escalable y portable

// program counter es global
int pc=0;


void ejecutaInstruccionA()
{
    std::cout<<"Estoy en la funcion A y el PC es: "<<pc<<"\n";
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); //dormir por 500 milisegundos el hilo de ejecucion
}


void ejecutaInstruccionB()
{
    std::cout<<"Estoy en la funcion B y el PC es: "<<pc<<"\n";
    pc++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); //dormir por 500 milisegundos el hilo de ejecucion
}





int main(int argc, char const *argv[])
{
    std::cout<<"\n------ Iniciando la ejecucion del programa ------\n\n";
    std::cout<<"PC --> "<<pc<<"\n";
    pc++;
    ejecutaInstruccionA();
    std::cout<<"Regresando el PC de la funcion A --> "<<pc<<"\n";
    pc++;
    ejecutaInstruccionB();
    std::cout<<"Regresando el PC de la funcion B --> "<<pc<<"\n";
    pc++;
    std::cout<<"PC al terminar la instruccion es --> "<<pc<<"\n";
    return 0;
}
