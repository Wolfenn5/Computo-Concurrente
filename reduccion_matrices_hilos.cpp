#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>


std::mutex mutex;



void reduccionMatriz(float*a,float*b,int filas,int columnas,int inicio,int fin)
{
    for(int i=inicio;i<fin;i++)
    {
        b[i]=0; // se inicializa en 0 por si hay basura ya que es un apuntador y usa memoria dinamica
        for(int j=0; j<columnas; j++)
        {
            int indice= i*columnas+j;
            std::lock_guard<std::mutex> lock (mutex); // se libera de forma semi-automatica el mutex y garantiza que los hilos sean concurrentes en vez de paralelos ya que se protege la matriz c al momento de modificarla
            {
                b[i]+= a[indice];
            }
        }
    }
}



int main(int argc, char const *argv[])
{
    int num_hilos= atoi(argv[1]); // parametro del main sera el numero de hilos


    int filas=3, columnas=3;
    //float a[filas*columnas], b[filas],
    // se cambio a apuntadores porque se pasan por referencia a sumaMatrices
    float*a=new float[filas*columnas];
    float*b=new float[filas]; // ahora b sera la matriz reducida

    

    // Inicializando la matriz (de 3x3 como valores de ejemplo en la practica)
    for(int i=0; i<filas*columnas; i++) // filas*cloumnas se puede hacer de una vez todo
    {
        a[i]= i+1; // 1, 2, 3 ... 9
    }

    // Imprimir matriz original
    std::cout<<"Matriz original:\n";
    for(int i=0; i<filas; i++)
    {
        for(int j=0; j<columnas; j++)
        {
            std::cout<<"["<<a[i*columnas+j]<<"]"<<" ";
        }
        std::cout<<"\n";
    }

    
    std::vector<std::thread> hilos;
    int filas_por_hilo=filas/num_hilos;



    auto inicio= std::chrono::high_resolution_clock::now(); // para medir el tiempo de ejecucion

    for(int i=0; i<num_hilos; i++)
    {
        // esta seccion se hace asi porque el numero de hilos se da desde el main
        // y sea posible distribuir bien las filas por cada hilo
        int inicio=i*filas_por_hilo;
        int fin=(i==num_hilos-1)?filas:(i+1)*filas_por_hilo;
        hilos.emplace_back(reduccionMatriz,a,b,filas,columnas,inicio,fin);
    }

    
    for(auto &hilo:hilos)
    {
        hilo.join();
    }

    auto fin= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo_ejecucion= fin-inicio;


    // Imprimir resultado
    std::cout<<"\nReduccion de la matriz:\n";
    for(int i=0;i<filas;i++)
    {
        std::cout<<"["<<b[i]<<"]";
    }
    std::cout<<"\n";


    std::cout<<"\n\nEl tiempo de ejecucion con "<<num_hilos<<" hilos es: "<<tiempo_ejecucion.count()<<"\n";
    delete[] a;
    delete[] b;

    return 0;
}
