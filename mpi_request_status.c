#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>



int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    MPI_Request request; // sirve para saber si una operacion finalizao o no
    MPI_Status status; //sirve para obtener informacion de un mensaje


    if (rank == 0)
    {
        int mensaje=42;
        MPI_Isend(&mensaje,1,MPI_INT,1,0,MPI_COMM_WORLD, &request);
        int a=5, b=2;
        printf("\nSoy el proceso %d y envie mi mensaje %d sin importar si ya lo recibieron\n",rank, mensaje);
        printf("\nLa suma de los valores que tengo es: %d\n",a+b);
    }
    else
    {
        int mensaje;
        MPI_Irecv(&mensaje,1,MPI_INT,0,0,MPI_COMM_WORLD,&request);
        int a=7, b=5;
        MPI_Wait(&request,&status); // si el wait va aqui siempre va a recibir el mensaje
        printf("\nSoy el proceso %d y no me importa si ya recibi el mensaje %d\n",rank, mensaje);
        printf("\nLa suma de los valores que tengo es: %d\n",a+b);
        //MPI_Wait(&request,&status); // si el wait va aqui puede que no reciba el mensaje
        printf("\nSoy el proceso %d y recibi el mensaje %d\n",rank,mensaje);
    }



    MPI_Finalize();
    return 0;
}
