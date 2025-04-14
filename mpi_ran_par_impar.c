#include <stdio.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc nombre_programa.c -o nombre_programa
mpirun -np 4 ./nombre_programa       con 4 procesos       ó
mpiexec -n 4 ./nombre_programa


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile lista_nodos.txt ./nombre_programa




Para subir archivos hay que escribir:

nano nombre_programa 
y pegar el contenido del programa, guardarlo y ya estando en un nodo se compila y ejecuta
*/

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Inicializa MPI

    int rank, size; // rank seria como un id del proceso
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Obtiene el ID del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Obtiene el número total de procesos

    if (rank %2 == 0)
    {
        printf("Soy  el proceso %d y soy par\n", rank);
    }
    else
    {
        printf("\nSoy el proceso %d y soy impar\n", rank);
    }
    

    MPI_Finalize();  // Finaliza MPI
    return 0;
}
