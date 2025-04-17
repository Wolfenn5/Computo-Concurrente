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


int main(int argc, char **argv)
{
    int rank, size, longitud;
    char nombre_host[MPI_MAX_PROCESSOR_NAME]; // variable donde se guarda el nombre del host con tamaño maximo
    char nombre_receptor[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(nombre_host, &longitud);


    // Como va a ser de tipo anillo, no se usa if else para que el proceso 0 solo envie ya que tambien va a recibir
    int siguiente= (rank+1) % size; // rank+1 por ser el siguiente proceso y con % se asegura que no envie a un proceso extra, si hay 4 procesos y se esta en el proceso 3 seria: (3+1)%4= 0
    int anterior= (rank-1+size) % size; // rank-1 para el proceso anterior y size para todos los procesos, si hay 4 procesos y se esta en el proceso 0 seria: (0-1+4)%4= 3

    // Enviar al siguiente proceso
    MPI_Send(nombre_host,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,siguiente,0,MPI_COMM_WORLD); //variable que se va a enviar, el tamaño de la variable, tipo caracter de MPI, identificador del proceso, etiqueta del mensaje


    // Recibir del proceso anterior
    MPI_Recv(nombre_receptor,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,anterior,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //donde se va a guardar,tamaño del mensaje, tipo de mensaje, de donde viene, etiqueta, comunicador, ignorar el estado porque en la misma funcion se tienen los datos de cada proceso

    printf("\nEl proceso %d recibio el mensaje %s desde el proceso %d\n",rank,nombre_receptor, anterior);



    MPI_Finalize();
    return 0;
}
