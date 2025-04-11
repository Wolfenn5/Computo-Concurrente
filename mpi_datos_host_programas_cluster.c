#include <stdio.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile /home/rober/lista_nodos.txt ./mpi
*/



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Inicializa MPI

    int rank, size;
    char nombre_proceso[MPI_MAX_PROCESSOR_NAME];

    int longitud_nombre;

    MPI_Get_processor_name(nombre_proceso, &longitud_nombre);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    printf("\nSoy el proceso %d de %d y me estoy ejecutando en el host %s\n", rank,size, nombre_proceso);
    MPI_Finalize();
    return 0;
}