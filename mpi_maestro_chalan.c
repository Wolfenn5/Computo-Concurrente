#include <stdio.h>
#include <mpi.h>
//#include <unistd.h>
#include <stdlib.h>

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



int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Parametro del main para indicar el tamaño del arreglo
    int N= atoi(argv[1]);
    int *arreglo;
    int base= N/(size-1);
    int sobrante= N%(size-1);

    if (rank == 0) // si el rank es 0 entonces es el maestro y debe designar trabajo
    {
        int valorReal=base;
        int suma=0;
        int sumaParcial=0;
        arreglo= (int *)malloc(sizeof(int)*N);
        for (int i=0; i<N; i++)
        {
            arreglo[i]= i+1;
        }

        int offset=0; // contador
        for (int i=0; i<size-1; i++)
        {
            if (i+1 <= sobrante)
            {
                valorReal= base+1;
            }            
            else
            {
                valorReal= base;
            }
            printf("\nMe tocan %d valores", valorReal);
            MPI_Send(arreglo + offset, valorReal, MPI_INT, i+1, 0, MPI_COMM_WORLD);
            offset+= valorReal; // actualizar el valor del contador
        }
        for (int i=0; i<size; i++)
        {
            MPI_Recv(&sumaParcial,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            suma+=sumaParcial;
        }
        printf("\nLa suma total es: %d\n",suma);
        
    }
    else // chalanes
    {
        int valorReal=base;
        int sumaParcial= 0;
        if (rank <= sobrante)
        {
            valorReal= base+1;
        }
        arreglo= (int *)malloc(sizeof(int)*valorReal);
        MPI_Recv(arreglo,valorReal,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("\nSoy el proceso %d y recibi:\n",rank);
        for (int i=0; i<valorReal; i++)
        {
            printf("%d ",arreglo[i]);
            sumaParcial+= arreglo[i];
        }
        printf("\n");
        printf("\nLa suma parcial de mis elementos es: %d\n",sumaParcial);
        MPI_Send(&sumaParcial,1,MPI_INT,0,0,MPI_COMM_WORLD);
    }


    MPI_Finalize();

    return 0;
}