#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[]) 
{
    // Parametro del main para el tamaño de la matriz
    int N= atoi(argv[1]);
    srand(time(NULL));


    // Declarar matrices en el host (CPU)
    float *matrizA_host= (float*) malloc(sizeof(float)*N*N);
    float *matrizB_host= (float*) malloc(sizeof(float)*N*N);
    float *matrizC_host= (float*) malloc(sizeof(float)*N*N);


    // Llenado de matrices
    for (int i=0; i<N*N; i++) 
    {
        matrizA_host[i]= (float) rand()/RAND_MAX; // valores entre 0 y 1
        //matrizA_host[i]= i+1; // valores consecutivos a partir de 1 (para probar)
    }

    for (int i=0; i<N*N; i++) 
    {
        matrizB_host[i]= (float) rand()/RAND_MAX; // valores entre 0 y 1
        //matrizB_host[i]= i+1; // valores consecutivos a partir de 1 (para probar)
    }



    // Medir el tiempo de ejecucion en la CPU
    clock_t inicio= clock(); // se marca en donde va a empezar a medir el tiempo

    // Multiplicación de matrices
    for (int i=0; i<N; i++) 
    {
        for (int j=0; j<N; j++) 
        {
            float suma= 0.0; // inicializar el elemento correspondiente al "arreglo" del resultado
            for (int k=0; k<N; k++) 
            {
                suma+= matrizA_host[i*N+k] * matrizB_host[k*N+j];
            }
            matrizC_host[i*N+j]= suma;
        }
    }
    clock_t fin= clock(); // se marca en donde va a terminar a medir el tiempo

 

    // Imprimir matriz C
    // printf("\nMatriz resultado:\n");
    // for (int i=0; i<N; i++) 
    // {
    //     for (int j=0; j<N; j++) 
    //     {
    //         printf("%f ", matrizC_host[i*N+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n%f\n", matrizC_host[(N*N)-1]); // ultimo elemento nadamas (para probar con valores grandes cuando se hace con numeros consecutivos)

    

    // Calcular el tiempo de ejecución en segundos
    double tiempo_CPU= ((double) (fin-inicio) / CLOCKS_PER_SEC) ;
    printf("\nEl tiempo de ejecucion en CPU fue de: %f segundos\n", tiempo_CPU);


    // Liberar memoria del host (CPU)
    free(matrizA_host);
    free(matrizB_host);
    free(matrizC_host);

    return 0;
}
