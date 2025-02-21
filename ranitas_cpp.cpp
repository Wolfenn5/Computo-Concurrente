#include <iostream>
#include <thread> //en vez de pthread y ademas es mas escalable
#include <vector> //para los hilos ranitas [ranas]
#include <cstdlib> //para usar srand

//bibliotecas del codigo program counter y dormir hilos
#include <ctime>   //para usar time
#include <chrono>  //para usar sleep_for



#define inicio 1
#define meta 300


//[nombre_programa, argv[1], argv[2]]
void brinca(int id) 
{
    int posicion= inicio;
    while(posicion<meta) 
    {
        posicion= posicion + (rand()%10+1); //valores entre 1 y 10
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); //dormir por 500 milisegundos el hilo de ejecucion
    }
    std::cout<<"\nHe llegado a la meta y soy la ranita "<<id<<"\n";
}


// ./ranitas_cpp 5   (lleva un argumento al ejecutar y es el numero de ranitas)
int main(int argc, char * argv []) //vamos a trabajar con parametros del main   
{
    srand(time(NULL)); //srand es de seed random time y NULL es que tomamos la hora de ejecucion del programa como semilla
    int ranas= atoi(argv[1]); // numero de ranas al ejecutar el codigo 
    std::cout<<"\nVoy a mandar a competir a "<<ranas<<" ranas\n";


    std::vector<std::thread> ranitas; // "arreglo" vector de hilos (ranas)


    //vamos a crear los hilos
    for(int i=0; i<ranas; i++)
    {
        ranitas.emplace_back(brinca,i); // es como usar push.back, pero el hilo se crea al momento de mandarlo al final del vector (std::thread)                 /*std::thread hilo_suma3(suma,(std::ref(numeros))); */
    }


    // esperamos a que todos los hilos terminen
    for(auto& cada_rana:ranitas)  // auto es un "tipo de dato" para trabajar con cualquier elemento de forma automatica
    {
        cada_rana.join();
    }



    return 0;
}

