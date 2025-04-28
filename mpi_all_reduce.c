#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define MAX_CITIES 52

typedef struct 
{
    int x[MAX_CITIES];
    int y[MAX_CITIES];
    int num_cities;
} instancia;



int leerInstancia(instancia data)
{
    FILE *fp = fopen("berlin52.tsp", "r");
    if(!fp){
        printf("No se pudo abrir el archivo.\n");
        return 1;
    }

    
    char line[128];
    while(fgets(line, sizeof(line), fp)){
        if(strncmp(line, "NODE_COORD_SECTION", 18) == 0)
            break;
    }

    int index;
    double x, y;
    data.num_cities = 0;
    while(fscanf(fp, "%d %lf %lf", &index, &x, &y) == 3){
        data.x[data.num_cities] = x;
        data.y[data.num_cities] = y;
        data.num_cities++;
    }
    fclose(fp);
   
}


void calculaDistancia(float matrizDistancia[MAX_CITIES][MAX_CITIES], instancia data)
{
    for (int i=0; i<MAX_CITIES; i++)
    {
        for (int j=0; j<MAX_CITIES; j++)
        {
            matrizDistancia[i][j]=sqrt(pow(data.x[i]-data.x[j],2)+pow(data.y[i]-data.y[j],2)); //raiz de (x2-x1)² + (y2-y1)² distancia euclidiana  
            printf("%f ",matrizDistancia[i][j]);
        }
        
    }
    
}


void generaTour(int arreglo[MAX_CITIES])
{
    for (int i=0; i<MAX_CITIES; i++)
    {
        int j= rand() % i+1; // aleatorios entre 0 y 1
        int temporal= arreglo[i];
        arreglo[i]= arreglo[j];
        arreglo[j]= temporal;
    }
    
}




int main(int argc, char *argv[])
{
    srand(time(NULL));
    MPI_Init (&argc, &argv);

    int rank,size;
    instancia data;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    leerInstancia(data); // son los datos del problema a resolver

    float matrizDistancia[MAX_CITIES][MAX_CITIES];

    calculaDistancia(matrizDistancia,data);
    int solucion[MAX_CITIES];
    generaTour(solucion);

    // Imprimir
    for (int i=0; i<MAX_CITIES; i++)
    {
        printf("%f-> ",solucion[i]);
    }
    for (int i=0; i<MAX_CITIES; i++)
    {
        for (int j=0; j<MAX_CITIES; j++)
        {
            printf("%d ",matrizDistancia[i][j]);
        }
        printf("\n");        
    }
    
    

    MPI_Finalize();
    return 0;
}
