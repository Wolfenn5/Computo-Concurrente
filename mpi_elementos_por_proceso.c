#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv); // iniciar mpi

    int n=atoi(argv[1]); // numero de elementos dado por el main

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int n_elementos_proceso= n/size; // dividir el numero de procesos entre el numero de elementos
    int restante= n%size; // por si la division no es exacta


    int inicio, elementos_hacer;
    if(rank<restante) // si la division de los elementos es impar
    {
        elementos_hacer= n_elementos_proceso+1; // es el numero de procesos +1 par asignar un elemento adicional
        printf("\nEl proceso %d y me tocan %d elementos\n", rank, elementos_hacer);
    }else
    {
        elementos_hacer= n_elementos_proceso; // no se añade un elemento adicional
        printf("\nEl proceso %d y me tocan %d elementos\n", rank, elementos_hacer);
    }

    
    


    MPI_Finalize(); // terminar mpi (como si fuera un join)
    return 0;
}
