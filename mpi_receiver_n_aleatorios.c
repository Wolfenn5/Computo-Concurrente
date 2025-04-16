#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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
    srand(time(NULL));
    int rank, size, longitud;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    // Si es el proceso 0 envia si no-(proceso 1), recibe
    if (rank == 0)
    {
        int aleatorio;
        for (int i=1; i<size; i++)
        {
            char nombre_host[MPI_MAX_PROCESSOR_NAME];
            int aleatorio= 1+rand()%10;
            MPI_Send(&aleatorio,MPI_MAX_PROCESSOR_NAME,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //donde se va a guardar,tamaño del mensaje, tipo de mensaje, de donde viene, etiqueta, comunicador, ignorar el estado porque en la misma funcion se tienen los datos de cada proceso
            printf("\nElnviamos el numero %d desde el proceso %d al proceso %d\n",aleatorio,rank,i);
        }
        // Una vez que se envia, hay que esperar a que todos tengan su calculo
        for (int i=0; i<size; i++)
        {
            MPI_Recv(&aleatorio,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            printf("\nRecibimos el mensaje con el numero %d con el numero %d",i,aleatorio);
        }
        
    }
    
    else
    {
        int aleatorio;
        char nombre_receptor[MPI_MAX_PROCESSOR_NAME]; // variable donde se guarda el nombre del host con tamaño maximo
        MPI_Recv(&aleatorio,MPI_MAX_PROCESSOR_NAME,MPI_INT,0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE); //variable que se va a enviar, el tamaño de la variable, tipo caracter de MPI, identificador del proceso, etiqueta del mensaje
        printf("\nEl proceso %s recibio el mensaje %d\n",rank,aleatorio);
        int cuadrado=pow(aleatorio,2);
        MPI_send(&cuadrado,1,MPI_INT,0,0,MPI_COMM_WORLD);
        printf("\nEnviamos el mensaje %d desde el proceso %d",cuadrado,rank);
    }

    MPI_Finalize();
    return 0;
}
