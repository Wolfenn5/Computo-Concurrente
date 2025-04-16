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
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    // Si es el proceso 0 envia si no-(proceso 1), recibe
    if (rank == 0)
    {
        char nombre_host[MPI_MAX_PROCESSOR_NAME]; // variable donde se guarda el nombre del host con tamaño maximo
        MPI_Get_processor_name(nombre_host,&longitud);
        for (int i=1; i<size; i++)
        {
            MPI_Send(&nombre_host,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,i,0,MPI_COMM_WORLD); //variable que se va a enviar, el tamaño de la variable, tipo caracter de MPI, identificador del proceso, etiqueta del mensaje
            printf("\nEnviamos el mensaje %s desde el proceso %d al proceso %d\n",nombre_host,rank,i);
        }
        
    }
    else
    {
        char nombre_receptor[MPI_MAX_PROCESSOR_NAME];
        MPI_Recv(&nombre_receptor,longitud,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //donde se va a guardar,tamaño del mensaje, tipo de mensaje, de donde viene, etiqueta, comunicador, ignorar el estado porque en la misma funcion se tienen los datos de cada proceso
        printf("\nEl proceso %d recibio el mensaje %s\n",rank,nombre_receptor);
    }

    MPI_Finalize();
    return 0;
}
