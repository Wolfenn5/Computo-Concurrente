#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

/* tamaños del vector 
1000
10000
100000
1000000
10000000
*/

int main(int argc, char const *argv[]) 
{
    //esta seccion se uso para saber como maneja los datos la computadora

	// printf("Tamano de int: %lu bytes\n", sizeof(int));                //4
    // printf("Tamano de long: %lu bytes\n", sizeof(long));              //4
    // printf("Tamano de long long: %lu bytes\n", sizeof(long long));    //8
    // printf("Tamano de float: %lu bytes\n", sizeof(float));            //4
    // printf("Tamano de double: %lu bytes\n", sizeof(double));          //8
    // printf("Tamano de char: %lu bytes\n", sizeof(char));              //1
    


    int tamanio_vector= atoi(argv[1]); // argumento de main para el tamaño del vector


    int *A= (int*)malloc(tamanio_vector*sizeof(int));
    int *B= (int*)malloc(tamanio_vector*sizeof(int));
   
    long long resultado=0;

    // vectores con valores de i (1,2,3,4.... hasta tamanio_vector)
    for(int i=0; i<tamanio_vector; i++)
    {
        A[i]= i+1;
        B[i]= tamanio_vector-i;
    }

    // Llenado de vectores (prueba con valores de la practica)
    // for(int i=0; i<5; i++)
    // {
    //     A[i]= i+1; // 1,2,3,4,5
    //     B[i]= i+6; // 6,7,8,9,10
    // }


    // Imprimir vector A
    // printf("\nEl vector A es:\n");
    // for (int i=0; i<tamanio_vector; i++)
    // {
    //     printf("[%d]", A[i]);
    // }

    // Imprimir vector B
    // printf("\n\nEl vector B es:\n");
    // for (int i=0; i<tamanio_vector; i++)
    // {
    //     printf("[%d]", B[i]);
    // }
    
    clock_t inicio, fin; // para saber el tiempo de ejecucion
    
    inicio= clock();
    for(int i=0;i<tamanio_vector;i++)
    {
        resultado+= (long long)A[i]*B[i]; // calculo del producto punto
    }
    fin= clock();

    double tiempo_ejecucion= (double)(fin - inicio) / CLOCKS_PER_SEC;

    printf("\n\nEl producto punto de A y B es: %lld y el tiempo de ejecucion es: %f\n",resultado, tiempo_ejecucion);

    return 0;
}

