#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>


void sumaVectores (float *A, float *B, float *C, int N)
{
    for (int i=0; i<N; i++)
    {
        C[i]= A[i]+B[i];
    }
    
}



int main(int argc, char const *argv[])
{
    srand(time(NULL));

    int N= atoi(argv[1]); // tamaño arreglo desde el main

    float *A=(float *) malloc(sizeof(float)*N);
    float *B=(float *) malloc(sizeof(float)*N);
    float *C=(float *) malloc(sizeof(float)*N);


    // Llenado de vectores
    for (int i=0; i<N; i++)
    {
        A[i]= (float)rand()/RAND_MAX;
        B[i]= (float)rand()/RAND_MAX;
        C[i]= (float)rand()/RAND_MAX;
    }
    

    clock_t inicio, fin;

    inicio= clock();
    sumaVectores(A,B,C,N);
    fin=clock();

    double tiempo_CPU= ((double) (fin-inicio) / CLOCKS_PER_SEC) ;

    printf("\nEl tiempo de ejecucion es de: %f\n",tiempo_CPU);
    return 0;
}
