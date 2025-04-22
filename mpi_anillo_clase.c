#include <stdio.h>
#include <mpi.h>
#include <math.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile /home/rober/lista_nodos.txt ./mpi

Para este programa incluir -lm para enlazar la biblioteca de math.h


Para subir archivos hay que escribir:

nano nombre_programa 
y pegar el contenido del programa, guardarlo y ya estando en un nodo se compila y ejecuta
*/

int esPotencia (int size)
{
    float N= log2(size);
    printf("\nValor de N=%f\n",N);

    if (floor(N)==N) // si la parte entera de N es = N
    {
        printf("\nSoy potencia de 2\n");
        return 1;
    }
    else
    {
        printf("\nNo soy potencia de 2\n");
        return 0;
    }
}



int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc,&argv);


    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Es necesario saber si el valor de N es 2^n es decir que n sea mayor o igual a 3
    int N=log2(size);
    //if ((N >= 2)  && (size%2==0)) // garantizar que haya al menos 4 procesos y que se tenga un numero de proceso par 
    if ((N >= 2)  && (esPotencia(size))) // garantizar quehaya al menos 4 procesos y que n sea potencia de 2 
    {
        char nombre[MPI_MAX_PROCESSOR_NAME];
        int longitud;
        MPI_Get_processor_name(nombre,&longitud);

        if ((rank%2) == 0) // si es par recibe
        {
            MPI_Recv(nombre,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            printf("\nSoy el proceso %d y recibi del proceso %d el mensaje %s\n",rank,rank+1,nombre);
        }
        else // si es impar envia
        {
            MPI_Send(nombre,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,rank-1,0,MPI_COMM_WORLD);
            printf("\nSoy el proceso %d y envie el mensaje al proceso %s %d\n",rank,nombre,rank-1);
        }
        MPI_Finalize();
    }
    else
    {
		printf("\nEl número de procesos debe ser par y >= 4\n");
		MPI_Finalize();
		return 1;
	}
    
    return 0;
}