#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

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

    printf("\nSoy el proceso %d antes de la barrera\n", rank);
    sleep(2);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("\nSoy el proceso %d despues de la barrera\n", rank);
    MPI_Finalize();

    return 0;
}