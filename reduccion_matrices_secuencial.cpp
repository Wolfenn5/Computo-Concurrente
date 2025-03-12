#include <iostream>
#include <thread>
#include <chrono>


int main(int argc, char const *argv[])
{
    int filas=10000, columnas=10000;
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
    // std::cout<<"Matriz original:\n";
    // for(int i=0; i<filas; i++)
    // {
    //     for(int j=0; j<columnas; j++)
    //     {
    //         std::cout<<"["<<a[i*columnas+j]<<"]"<<" ";
    //     }
    //     std::cout<<"\n";
    // }
 

    auto inicio= std::chrono::high_resolution_clock::now(); // para medir el tiempo de ejecucion

    for (int i=0; i<filas; i++) 
    {
        b[i] = 0; // se inicializa en 0 por si hay basura ya que es un apuntador y usa memoria dinamica
        for (int j=0; j<columnas; j++) 
        {
            int indice= i*columnas+j;
            b[i] += a[indice]; // Acumulando la suma de la fila
        }
    }
    auto fin= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo_ejecucion= fin-inicio;


    // Imprimir resultado
    // std::cout<<"\nReduccion de la matriz:\n";
    // for(int i=0;i<filas;i++)
    // {
    //     std::cout<<"["<<b[i]<<"]";
    // }
    // std::cout<<"\n";


    std::cout<<"\n\nEl tiempo de ejecucion es: "<<tiempo_ejecucion.count()<<"\n";
    delete[] a;
    delete[] b;

    return 0;
}
