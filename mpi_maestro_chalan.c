#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile /home/rober/lista_nodos.txt ./mpi

Para este programa incluir -lm para enlazar la biblioteca de math.h


Para subir archivos hay que escribir:

nano nombre_programa 
y pegar el contenido del programa, guardarlo y ya estando en un nodo se compila y ejecuta
*/



int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Parametro del main para indicar el tamaño del arreglo
    int N= atoi(argv[1]);
    int arreglo[N];

    if (size == 3) //que solo funcione con 3 procesos
    {
        if (rank == 0) // si el rank es 0 entonces es el maestro y debe designar trabajo
        {
            for (int i=0; i<N; i++)
            {
                arreglo[i]= i+1;
            }
            MPI_Send(arreglo+0,N/2,MPI_INT,1,0,MPI_COMM_WORLD);
            MPI_Send(arreglo+N,N/2,MPI_INT,1,0,MPI_COMM_WORLD);
        }
        else
        {
            int subarreglo[N/2];
            MPI_Recv(subarreglo,N/2,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            printf("\nSoy el proceso %d y recibi:\n",rank);
            for (int i=0; i<N/2; i++)
            {
                printf("%d ",subarreglo[i]);
            }
            printf("\n");
        }
    }



    MPI_Finalize();

    return 0;
}