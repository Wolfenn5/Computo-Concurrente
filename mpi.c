#include <stdio.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mpi.c -o hola_mpi
mpirun -np 4 ./hola_mpi       con 4 procesos

*/

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Inicializa MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Obtiene el ID del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Obtiene el número total de procesos

    printf("Hola mundo desde el proceso %d de %d\n", rank, size);

    MPI_Finalize();  // Finaliza MPI
    return 0;
}