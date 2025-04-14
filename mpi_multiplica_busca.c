#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Compilacion y ejecucion del programa que utilza mpi

mpicc hola_mundo_mpi.c -o hola_mundo_mpi
mpirun -np 4 ./mpi       con 4 procesos       ó
mpiexec -n 4 ./mpi


Con el lcluster y la lista de nodos en el /home/rober/lista_nodos.txt
mpiexec -n 4 --hostfile /home/rober/lista_nodos.txt ./mpi
*/



int main(int argc, char**argv) 
{
    MPI_Init(&argc, &argv);  // Inicializa MPI

    int rank, size;

    //Parametro del main tamaño del arreglo
    int n= atoi(argv[1]); 



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

    int valor_local= rank;
    // Si el color (comunicador o fila_id) es par entonces va a hacer la multiplicacion de su rank*2 y luego se van a sumar y si es impar va a hacer la busqueda sobre un arreglo aleatorio

    if (fila_id%2 == 0) // si es par
    {
        valor_local= rank*2;
        printf("\nSoy el proceso %d y mi multiplicacion es %d\n",rank, valor_local);
        int suma;
        MPI_Reduce(
            &valor_local, // que cosa se toma
            &suma, // en donde se almacena
            1, // el numero de elementos que va a tener suma
            MPI_INT,
            MPI_SUM,
            0, // se le asignara al proceso 0 del comunicador de fila
            comunicador_fila // sobre el comunicador de fila
        );
        if (rank_local == 0)
        {
            printf("\nLa suma de las multiplicaciones de los ranks en el comunicador %d es: %d\n",fila_id, suma);
        }
        
    }
    else
    {
        int arreglo[n];
        int encontrado=0;
        for (int i=0; i<n; i++)
        {
            arreglo[i]= rand() % n; // valores del arreglo aleatorios entre 0 y n-1
        }

        for (int i=0; i<n; i++)
        {
            if (rank == arreglo[i])
            {
                encontrado=1;
                printf("\nSoy el proceso %d y me encontre en el arrelo",rank_local);
                break;
            }
            
        }
        int totales;
        MPI_Reduce(
            &encontrado, // que cosa se toma
            &totales, // en donde se almacena
            1, // el numero de elementos que va a tener suma
            MPI_INT,
            MPI_SUM,
            0, // se le asignara al proceso 0 del comunicador de fila
            comunicador_fila // sobre el comunicador de fila
        );
        if (rank_local == 0)
        {
            printf("\nSomos el grupo B y encontramos %d veces nuestro rank en el arreglo %d\n",totales,fila_id);
        }
        
    }
    

    MPI_Finalize();
    return 0;
}