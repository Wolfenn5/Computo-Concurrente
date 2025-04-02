#include <iostream>
#include <chrono>



int main(int argc, char const *argv[])
{
    // Parametros del main 
    int filas= atoi(argv[1]);
    int columnas= atoi(argv[2]);



    // Declarar matrices
    float *a= new float[filas*columnas];
    float *b= new float[filas*columnas];
    float *c= new float[filas*columnas];


    // Inicializando matrices...
    for (int i=0; i<filas*columnas; i++) 
    {
        a[i]= i+1;  // valores de 1,2,3 ...
        b[i]= (filas*columnas)-i;  // valores de 16 ... 3,2,1
    }


    auto inicio = std::chrono::high_resolution_clock::now(); // marcar el inicio del tiempo a medir
    // Suma de matrices 
    for (int i=0; i<filas; i++) 
    {
        for (int j=0; j<columnas; j++) 
        {
            int indice= i*columnas+j;
            c[indice] = a[indice] + b[indice];
        }
    }
    auto fin = std::chrono::high_resolution_clock::now(); // marcar el final del tiempo a medir


    // Imprimir matriz a
    std::cout<<"Matriz a:\n";
    for(int i=0; i<filas; i++)
    {
        for(int j=0; j<columnas; j++)
        {
            std::cout<<"["<<a[i*columnas+j]<<"]"<<" ";
        }
        std::cout<<"\n";
    }

    // Imprimir matriz b
    std::cout<<"\n\nMatriz b:\n";
    for(int i=0; i<filas; i++)
    {
        for(int j=0; j<columnas; j++)
        {
            std::cout<<"["<<b[i*columnas+j]<<"]"<<" ";
        }
        std::cout<<"\n";
    }


    // Imprimir matriz c
    std::cout<<"\nLa suma de las matrices A y B es:\n";
    for(int i=0;i<filas;i++)
    {
        for(int j=0; j<columnas; j++)
        {
            std::cout<<"["<<c[i*columnas+j]<<"]";
        }
        std::cout<<"\n";
    }


    
    std::chrono::duration<double> tiempo_ejecucion = fin - inicio; // calcular el tiempo de ejecucion
    std::cout << "\nEl tiempo de ejecucion secuencial es: " << tiempo_ejecucion.count() << " segundos\n";

    // Liberar recursos de las matrices
    delete[] a;
    delete[] b;
    delete[] c;
    
    return 0;
}







