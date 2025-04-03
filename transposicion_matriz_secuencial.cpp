#include <iostream>
#include <chrono>



int main(int argc, char const *argv[])
{
    srand(time(NULL));


    // Parametro del main 
    // --> el primer parametro es N (que resultara en el tamaño de las matrices NxN)
    int filas= atoi(argv[1]);
    int columnas= filas;



    //float a[filas*columnas], b[filas*columnas], c[filas*columnas];
    // se cambio a apuntadores porque se pasan por referencia a transponerMatriz
    float *a= new float[filas*columnas]; // matriz a
    float *aTranspuesta= new float[filas*columnas]; // matriz a transpuesta



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



    auto inicio= std::chrono::high_resolution_clock::now(); // para medir el tiempo de ejecucion
    // Transposicion de matriz
    for (int i=0; i<filas; i++)
    {
        for (int j=0; j<columnas; j++)
        {
            int indice_matrizA= i*columnas+j;
            int indice_matrizTranspuesta= j*filas+i;
            {
                aTranspuesta[indice_matrizTranspuesta]= a[indice_matrizA]; // los indices hacen la transposición -->  A[i][j]    T[j][i]
            }
        }
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

    std::cout<<"\n\nEl tiempo de ejecucion es: "<<tiempo_ejecucion.count()<<" segundos\n";

    return 0;
}
