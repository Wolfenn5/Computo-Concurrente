#include <stdio.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile /home/rober/lista_nodos.txt ./mpi




Para subir archivos hay que escribir:

nano nombre_programa 
y pegar el contenido del programa, guardarlo y ya estando en un nodo se compila y ejecuta
*/



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Inicializa MPI

    int rank, size;
    char nombre_proceso[MPI_MAX_PROCESSOR_NAME];

    int longitud_nombre;

    MPI_Get_processor_name(nombre_proceso, &longitud_nombre);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Se va a asumir que los bloques van a ser de 4 elementos
    int columnas=4; // trabajar con 4 elementos
    int fila_id=rank/columnas;  // 0/4=0  5/4=1   8/4=2

    MPI_Comm comunicador_fila; // es un tipo de dato nativo que sirve para definir comunicadores locales

    MPI_Comm_split(MPI_COMM_WORLD, fila_id, rank, &comunicador_fila);

    printf("\nSoy el proceso %d de %d y me estoy ejecutando en el host %s\n", rank,size, nombre_proceso);
    int rank_local, size_local;

    MPI_Comm_rank(comunicador_fila, &rank_local);
    MPI_Comm_size(comunicador_fila, &size_local);

    printf("\nSoy el proceso %d de %d y me estoy ejecutando en el comunicador %s\n", rank_local,size_local, fila_id);
    MPI_Finalize();
    return 0;
}