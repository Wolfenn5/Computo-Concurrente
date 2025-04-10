#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>



__device__ void imprimeVectorDevice (int*d)
{
    printf("%d ",d[blockIdx.x * blockDim.x + threadIdx.x]); 
}


__global__ void reverse_dinamico(int *d, int n){
	extern __shared__ int s[]; //la directiva shared nos ayuda a especificar que la variable va a ser almacenada en el Streaming Multiprocessor (bloque)
	int t = blockIdx.x*blockDim.x+threadIdx.x;
	int local = threadIdx.x;
	//printf("\n Id global %d y el Id local %d\n",t,local);
	if(t < n){
		//int tr = n - t -1; //la posicion contraria a "t"
		//copiamos desde el device (RAM de la GPU) hacia el SM
		s[local]=d[t];
		__syncthreads();//esperamos a que cada uno de los hilos termine de acceder a la memoria compartida
		d[t]=s[blockDim.x-local-1];//copiamos cruzado
		//imprimeVectorDevice(d);
	}
}




int main(int argc, char const *argv[])
{
    // Parametro del main para tamaño
    int n= atoi(argv[1]);

    int *arreglo_host=(int*)malloc(sizeof(int)*n);  // Esta variable se aloja en la RAM del host

    int*arreglo_dispositivo; // Esta variable se aloja en la RAM del dispositivo

    for (int i=0; i<n; i++)
    {
        arreglo_host[i]= i+1;
        printf("%d ", arreglo_host[i]);
    }
    printf("\n");

    
    cudaMalloc(&arreglo_dispositivo,sizeof(int)*n);
    cudaMemcpy(arreglo_dispositivo, arreglo_host,sizeof(int)*n,cudaMemcpyHostToDevice);

    cudaDeviceProp propiedades;
    cudaGetDeviceProperties (&propiedades,0);


    int numeroHilos= propiedades.maxThreadsPerBlock;
    int numeroBloques= (n+numeroHilos-1) / numeroHilos;
    int tamanioShared= sizeof(int)*n; // tamaño de la memoria que se va a pedir 

    reverse_dinamico<<<numeroBloques, numeroHilos, tamanioShared>>>(arreglo_dispositivo,n);
    cudaDeviceSynchronize();


    cudaMemcpy(arreglo_host,arreglo_dispositivo,sizeof(int)*n, cudaMemcpyDeviceToHost);


    cudaFree(arreglo_dispositivo);
    return 0;
}
