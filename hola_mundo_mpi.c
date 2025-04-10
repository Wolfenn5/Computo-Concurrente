#include <stdio.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi

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