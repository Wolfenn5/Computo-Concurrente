#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

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


__global__ void MatrizxVector (float * matriz, float * arreglo, float * resultado, int n, int m)
{
    int id_hilo= blockIdx.x * blockDim.x + threadIdx.x; // bloque 0 dimension bloque 512 + id hilo 0 seria el hilo 512
    
    if (id_hilo<n) // si el id hilo es menor al numero total de datos (filas) hacer el procesamiento
    {
        printf("\nblockIdx: %d",blockIdx.x);
        printf("\nblockDimx: %d",blockDim.x);
        printf("\nthreadIdx: %d",threadIdx.x);
        resultado[id_hilo]=0.0; // inicializar el elemento correspondiente al arreglo del resultado
    }


        for (int i=0; i<m; i++)
        {
            // calcular el indice porque la matriz se esta trabajando como un arreglo (doble apuntador)
            // se usa el numero de columnas como base para el desplazamiento (i*m+j)
            resultado[id_hilo]+= (matriz[id_hilo*m+i]) * (arreglo[i]);
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


    float * matriz_host=(float*) malloc(sizeof(float)*n*m);
    float * arreglo_host=(float*) malloc(sizeof(float)*m);
    float * resultado_host=(float*) malloc(sizeof(float)*n);

    // declarar las variables en el dispositivo
    float *matriz_dispositivo, *arreglo_dispositivo, *resultado_dispositivo;
    cudaMalloc(&matriz_dispositivo,n*m*sizeof(float));
    cudaMalloc(&arreglo_dispositivo,m*sizeof(float));
    cudaMalloc(&resultado_dispositivo,n*sizeof(float));


    for (int i=0; i<n*m; i++)
    {
        matriz_host[i]=(float) rand()/RAND_MAX; // valores entre 0 y 1
    }

    for (int i=0; i<n*m; i++)
    {
        arreglo_host[i]=(float) rand()/RAND_MAX; // valores entre 0 y 1
    }
    

    // Copiar del host al dispositivo
    // destino, origen, tamaño, confirmacion
    cudaMemcpy(matriz_dispositivo, matriz_host, sizeof(float)*n*m, cudaMemcpyHostToDevice); 
    cudaMemcpy(arreglo_dispositivo, arreglo_host, sizeof(float)*m, cudaMemcpyHostToDevice);


    // Calcular el numero de hilos a ocupar
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades,0);
    int tamanio_bloque= propiedades.maxThreadsPerBlock; 
    int numero_bloques= ((n+tamanio_bloque-1)/tamanio_bloque); // n es el numero de datos, la formula es universal
    // Si se quiere saber el numero de hilos maximo se multiplica tamanio_bloque*numero_bloques


    MatrizxVector<<<numero_bloques, tamanio_bloque>>>(matriz_dispositivo, arreglo_dispositivo, resultado_dispositivo, n,m);
    cudaMemcpy(resultado_host, resultado_dispositivo,sizeof(float)*n,cudaMemcpyDeviceToHost);
    printf("\nEl vector resultante es:\n");
    for (int i=0; i<n; i++)
    {
        printf("%f ",resultado_host[i]);
    }
    printf("\n");
    
    // Liberar recursos del dispositivo (GPU)
    cudaFree(matriz_dispositivo);
    cudaFree(arreglo_dispositivo);
    cudaFree(resultado_dispositivo);

    // Liberar recursos del host (CPU)
    free (matriz_host);
    free(arreglo_host);
    free(resultado_host);

    return 0;
}
