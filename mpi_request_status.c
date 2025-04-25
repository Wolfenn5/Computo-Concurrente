#include <stdio.h>
#include <mpi.h>
//para el envio y recepcion de mensajes de forma asincrona
int main(int argc, char * argv[]){
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Request request;//nos sirve para saber si finalizó o no cierta operación
	MPI_Status status;//nos sirve para obtener la información sobre el mensaje
	if(rank==0){
		int mensaje=42;
        // MPI_Send(&mensaje,1,MPI_INT,1,0,MPI_COMM_WORLD); // aqui mpi seria sincrono
		MPI_Isend(&mensaje,1,MPI_INT,1,0,MPI_COMM_WORLD, &request);
		int a=5, b=2;
		printf("\nSoy el proceso %d y envie mi mensaje %d y no me importa si ya lo recibieron\n",rank,mensaje);
		printf("\nLa suma de los valores que tengo es: %d\n",a+b);
	}
	else{
		int mensaje;
		MPI_Irecv(&mensaje,1,MPI_INT,0,0,MPI_COMM_WORLD, &request);
        // MPI_Recv(&mensaje,1,MPI_INT,0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE); // aqui mpi seria sincrono
		int a=7, b=5;
		printf("\nSoy el proceso %d y no me importa si ya recibi el mensaje %d\n",rank,mensaje);
		printf("\nLa suma de los valores que tengo es: %d\n",a+b);
		MPI_Wait(&request,&status); // aqui mpi seria asincrono
		printf("\nSoy el proceso %d y recibí el mensaje %d\n",rank,mensaje);
	}
	MPI_Finalize();
	return 0;
}