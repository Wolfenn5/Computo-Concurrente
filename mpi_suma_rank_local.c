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

    // Se va a asumir que los bloques van a ser de 4 elementos
    int columnas=4; // trabajar con 4 elementos
    int fila_id=rank/columnas;  // 0/4=0  5/4=1   8/4=2

    MPI_Comm comunicador_fila; // es un tipo de dato nativo que sirve para definir comunicadores locales

    MPI_Comm_split(MPI_COMM_WORLD, fila_id, rank, &comunicador_fila);

    printf("\nSoy el proceso %d de %d y me estoy ejecutando en el host %s\n", rank,size, nombre_proceso);
    int rank_local, size_local;

    MPI_Comm_rank(comunicador_fila, &rank_local);
    MPI_Comm_size(comunicador_fila, &size_local);

    printf("\nSoy el proceso %d de %d y me estoy ejecutando en el comunicador %d\n", rank_local,size_local, fila_id);




    // reduce permite ejecutar funciones en los procesos y que se envie el resultado a un proceso especifico
    // para este caso reduce ira al proceso 0 (local)
    int valor_local= rank_local;
    int suma;
    MPI_Reduce(&valor_local, // valor que se va a sumar por cada proceso
        &suma, // valor que se va a guardar
        1, // guardar un elemento (tamaño de la suma)
        MPI_INT, // el tipo de dato a trabaar es de tipo entero
        MPI_SUM, // el tipo de operacion de reduccion que se va a hacer, en este caso la suma de vlaores locales
        0, // solo el proceso 0 se va a imprimir
        comunicador_fila // comunicador de procesos
    );



    if (rank_local==0)
    {
        printf("\nLa suma de los IDs de mis amigos de la fila %d es: %d\n", fila_id, suma);
    }
    
    
    MPI_Finalize();
    return 0;
}