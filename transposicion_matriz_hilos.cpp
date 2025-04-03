#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>


std::mutex mutex;



void transponerMatriz(float *a, float *t, int filas, int columnas, int inicio, int fin)
{
    for (int i=inicio; i<fin; i++)
    {
        for (int j=0; j<columnas; j++)
        {
            int indice_matrizA= i*columnas+j;
            int indice_matrizTranspuesta= j*filas+i;
            std::lock_guard<std::mutex> lock (mutex); // se libera de forma semi-automatica el mutex y garantiza que los hilos sean concurrentes en vez de paralelos ya que se protege la matriz c al momento de modificarla
            {
                t[indice_matrizTranspuesta]= a[indice_matrizA]; // los indices hacen la transposición -->  A[i][j]    T[j][i]
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
    int num_hilos= atoi(argv[2]);



    //float a[filas*columnas], b[filas*columnas], c[filas*columnas];
    // se cambio a apuntadores porque se pasan por referencia a transponerMatriz
    float *a= new float[filas * columnas]; // matriz a
    float *aTranspuesta= new float[filas * columnas]; // matriz a transpuesta



    // Inicializar matriz A 
    for (int i=0; i<filas*columnas; i++)
    {
        //a[i]= i+1; // valores de 1,2,3 ... 16 (para probar)  
        a[i]= (float) rand()/RAND_MAX; // aleatorios entre 0 y 1
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


    std::vector<std::thread> hilos;
    int filas_por_hilo= filas/num_hilos;




    auto inicio= std::chrono::high_resolution_clock::now(); // para medir el tiempo de ejecucion

    for (int i=0; i<num_hilos; i++)
    {
        int inicio= i*filas_por_hilo;
        int fin= (i == num_hilos-1) ? filas: (i+1) * filas_por_hilo;
        hilos.emplace_back(transponerMatriz, a, aTranspuesta, filas, columnas, inicio, fin);
    }


    for (auto &hilo : hilos)
    {
        hilo.join();
    }

    auto fin= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo_ejecucion= fin-inicio;


    // Imprimir matriz a transpuesta
    // std::cout<<"\nLa matriz transpuesta es:\n";
    // for(int i=0;i<filas;i++)
    // {
    //     for(int j=0; j<columnas; j++)
    //     {
    //         std::cout<<"["<<aTranspuesta[i*columnas+j]<<"]";
    //     }
    //     std::cout<<"\n";
    // }

    std::cout<<"\n\nEl tiempo de ejecucion con "<<num_hilos<<" hilos es: "<<tiempo_ejecucion.count()<<" segundos\n";

    return 0;
}
