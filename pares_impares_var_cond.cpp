#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable> // biblioteca para variables de condicion

/* El programa hace una suma del 1 al 100 de pares e impares con un hilo para cada tipo de suma*/
// notify_one es equivalente a cond_signal
// notify_all es equivalente a cond_broadcast



std::mutex mutex;
std::condition_variable variable_condicion;



int suma_pares=0;
int suma_impares=0;
int numero_par=0;
int numero_impar=0;
bool par=false, impar= false, terminado_pares=0, terminado_impares=0; // banderas para la impresion


void calculo_pares()
{
    for (int i=2; i<=100; i=i+2) // sumar desde 2 hasta 100 solo numeros pares
    {
        {
            std::lock_guard<std::mutex> lock (mutex); // sirve para liberar de forma "semi-automatica" el mutex en vez de usar lock y unlock
            numero_par= i;
            suma_pares= suma_pares + numero_par;
            par= true;
        }
        variable_condicion.notify_one(); // notificar al hilo de impresion que modificamos algo
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        terminado_pares=true;
    }
    variable_condicion.notify_one(); // notificar al hilo de impresion que modificamos algo
}



void calculo_impares()
{
    for (int i=1; i<=100; i=i+2) // sumar desde 2 hasta 100 solo numeros impares
    {
        {
            std::lock_guard<std::mutex> lock (mutex); // sirve para liberar de forma "semi-automatica" el mutex en vez de usar lock y unlock
            numero_impar= i;
            suma_impares= suma_impares + numero_impar;
            impar= true;
        }
        variable_condicion.notify_one(); // notificar al hilo de impresion que modificamos algo
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        terminado_impares=true;
    }
    variable_condicion.notify_one(); // notificar al hilo de impresion que modificamos algo
}


void imprimir()
{
    while (true) // El hilo imprimir siempre va a estar escuchando
    {
        //std::lock_guard<std::mutex> lock (mutex);
        std::unique_lock<std::mutex> lock (mutex); // unique lock sirve para bloquear y desbloquear varios mutex
        variable_condicion.wait(lock, [] { return par || impar || (terminado_pares && terminado_impares); }); // wait se queda esperando y hace que el mutex se libere de forma temporal. sus argumentos son std::lock y luego se dice que va a hacer cuando se despierte el hilo
        // ademas una vez que termine de escuchar, se vuelve a bloquear
        
        
        
        if (par) // si el numero es par se imprime la suma de los pares
        {
            std::cout<<"Numero agregado a la suma de pares"<<" y la suma total es: "<<suma_pares<<"\n";
            par=false;
        }
        
        
        if (impar) // si el numero es impar se imprime la suma de los impares
        {
            std::cout<<"Numero agregado a la suma de impares"<<" y la suma total es: "<<suma_impares<<"\n";
            impar=false;
        }
        
        
    
        if (terminado_pares && terminado_impares) // si ambos hilos terminan, imprimir el resultado final
        {
            std::cout<<"Ambos hilos terminaron y la suma par es: "<<suma_pares<<" y la suma impar es "<<suma_impares<<"\n";
            break; // para romper el while(true)
        }
        
    }
       
}









int main(int argc, char const *argv[])
{
    // Declarar hilos
    std::thread hilo_pares(calculo_pares);
    std::thread hilo_impares(calculo_impares);
    std::thread hilo_impresion(imprimir);


    hilo_pares.join();
    hilo_impares.join();
    hilo_impresion.join();


    return 0;
}
