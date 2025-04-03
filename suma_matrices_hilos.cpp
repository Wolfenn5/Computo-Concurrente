#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>


std::mutex mutex;



void sumaMatrices (float *a, float *b, float *c, int filas, int columnas, int inicio, int fin)
{
    for (int i=inicio; i<fin; i++)
    {
        for (int j=0; j<columnas; j++)
        {
            int indice= i*columnas+j;
            std::lock_guard<std::mutex> lock (mutex); // se libera de forma semi-automatica el mutex y garantiza que los hilos sean concurrentes en vez de paralelos ya que se protege la matriz c al momento de modificarla
            {
                c[indice]= a[indice] + b[indice];
            }
            
        }
        
    }
}



int main(int argc, char const *argv[])
{
    srand(time(NULL));


    // Parametros del main 
    // --> el primer parametro es N (que resultara en el tamaño de las matrices NxN) y el 2do es el numero de hilos
    int filas= atoi(argv[1]);
    int columnas= filas;
    int num_hilos=atoi(argv[2]);
    


    //float a[filas*columnas], b[filas*columnas], c[filas*columnas];
    // se cambio a apuntadores porque se pasan por referencia a sumaMatrices
    float*a=new float[filas*columnas];
    float*b=new float[filas*columnas];
    float*c=new float[filas*columnas];


    // Inicializando matrices...
    for(int i=0; i<filas*columnas; i++) // filas*cloumnas se puede hacer de una vez todo
    {
        //a[i]= i+1; // valores de 1,2,3 ... 16 (para probar)   
        //b[i]= (filas*columnas)-i; // valores de 16 ... 3,2,1 (para probar) 
        a[i]= (float) rand()/RAND_MAX; // aleatorios entre 0 y 1
        b[i]= (float) rand()/RAND_MAX; // aleatorios entre 0 y 1
    }

    // Imprimir matriz a
    // std::cout<<"Matriz a:\n";
    // for(int i=0; i<filas; i++)
    // {
    //     for(int j=0; j<columnas; j++)
    //     {
    //         std::cout<<"["<<a[i*columnas+j]<<"]"<<" ";
    //     }
    //     std::cout<<"\n";
    // }

    // Imprimir matriz b
    // std::cout<<"\n\nMatriz b:\n";
    // for(int i=0; i<filas; i++)
    // {
    //     for(int j=0; j<columnas; j++)
    //     {
    //         std::cout<<"["<<b[i*columnas+j]<<"]"<<" ";
    //     }
    //     std::cout<<"\n";
    // }
    
    
    std::vector<std::thread> hilos;
    int filas_por_hilo= filas/num_hilos;




    auto inicio= std::chrono::high_resolution_clock::now(); // para medir el tiempo de ejecucion

    for (int i=0; i<num_hilos; i++)
    {
        int inicio= i*filas_por_hilo;
        int fin= (i == num_hilos-1) ? filas: (i+1) * filas_por_hilo;
        hilos.emplace_back(sumaMatrices, a, b, c, filas, columnas, inicio, fin);
    }
    

    for (auto &hilo : hilos)
    {
        hilo.join();
    }

    auto fin= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo_ejecucion= fin-inicio;


    // Imprimir matriz c
    // std::cout<<"\nLa suma de las matrices A y B es:\n";
    // for(int i=0;i<filas;i++)
    // {
    //     for(int j=0; j<columnas; j++)
    //     {
    //         std::cout<<"["<<c[i*columnas+j]<<"]";
    //     }
    //     std::cout<<"\n";
    // }

    std::cout<<"\n\nEl tiempo de ejecucion con "<<num_hilos<<" hilos es: "<<tiempo_ejecucion.count()<<"\n";
    
    return 0;
}



