#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*Version secuencial (hay version de cuda con el mismo nombre vector_matriz.cu)*/


/*
Matriz A = [
1 2 3
4 5 6 
7 8 9
]

Arreglo 1 = [
             1
             2
             3
]


Arreglo Resultante = [
                       1*1 + 2*2 + 3*3
                       4*1 + 5*2 + 6*3
                       7*1 + 8*2 + 9*3
]

*/


void MatrizxVector (float * matriz, float * arreglo, float * resultado, int n, int m)
{
    for (int i=0; i<n; i++)
    {
        resultado[i]=0; // inicializar el elemento correspondiente al arreglo del resultado

        for (int j=0; j<m; j++)
        {
            // calcular el indice porque la matriz se esta trabajando como un arreglo (doble apuntador)
            // se usa el numero de columnas como base para el desplazamiento (i*m+j)
            resultado[i]+= (matriz[i*m+j]) * (arreglo[j]);
        }
        
    }
    
}


int main(int argc, char const *argv[])
{
    // para que se pueda multiplicar un vector por una matriz, las columnas deben coincidir con el numero de elementos del arreglo
    // una matriz de n*m puede multiplicarse por un arreglo de m*1

    srand(time(NULL));
    // Parametros del main filas y columnas 
    int n= atoi(argv[1]);
    int m= atoi(argv[2]);


    float ** matriz=(float**) malloc(sizeof(float)*n*m);
    float * arreglo=(float*) malloc(sizeof(float)*m);
    float * resultado=(float*) malloc(sizeof(float)*n);

    for (int i=0; i<n*m; i++)
    {
        matriz[i]=(float) rand()/RAND_MAX; // valores entre 0 y 1
    }

    for (int i=0; i<n*m; i++)
    {
        arreglo[i]=(float) rand()/RAND_MAX; // valores entre 0 y 1
    }
    
    MatrizxVector(matriz, arreglo, resultado, n,m);
    printf("\nEl vector resultante es:\n");
    for (int i=0; i<n; i++)
    {
        printf("%f ",resultado[i]);
    }
    printf("\n");
    
    return 0;
}
