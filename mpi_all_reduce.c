#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define MAX_CITIES 52

typedef struct {
	double x[MAX_CITIES];
	double y[MAX_CITIES];
	int num_cities;

} instancia;
int leerInstancia(instancia *data){
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
    data->num_cities = 0;
    while(fscanf(fp, "%d %lf %lf", &index, &x, &y) == 3){
        data->x[data->num_cities] = x;
        data->y[data->num_cities] = y;
        data->num_cities++;

    }
    printf("\nNúmero de ciudades: %d\n",data->num_cities);
    fclose(fp);
    
}

void calcularDistancia(double matrizDistancia[MAX_CITIES][MAX_CITIES], instancia data){
	for(int i=0;i<MAX_CITIES;++i)//distancia euclidiana o distancia entre 2 puntos
		for(int j=0;j<MAX_CITIES;++j){
			matrizDistancia[i][j]=sqrt(pow(data.x[j]-data.x[i],2)+pow(data.y[j]-data.y[i],2));//raiz((x2-x1)^2+(y2-y1)^2)
		}

}

//1 2 3 4 5  
void generaTour(int arreglo[MAX_CITIES]){
	for(int i=0;i<MAX_CITIES;++i)
		arreglo[i]=i+1;
	for(int i=0;i<MAX_CITIES;++i){
		int j = rand() % MAX_CITIES;
		int temporal = arreglo[i];
		arreglo[i]=arreglo[j];
		arreglo[j]=temporal;
	}
	for(int i=0;i<MAX_CITIES;++i)
		printf("%d->",arreglo[i]);
	printf("%d\n",arreglo[0]);
}


double evaluar_tour(int *tour, double matrizDistancia[MAX_CITIES][MAX_CITIES]) // recibe un arreglo con el orden de visita de los lugares (tour) y la matriz de distancia para poder sumar el valor etre cada ciudad
{
	double costo= 0.0; // variable para sumar los costos de los caminos que existen en el tour
	for (int i=0; i<MAX_CITIES-1; i++)
	{
		costo+= matrizDistancia[tour[i]][tour[i+1]];
	}
	costo+=matrizDistancia[tour[MAX_CITIES-1]][tour[0]];
	return costo;
}


void generaVecino (int * tour)
{
	int i= rand() % MAX_CITIES;
	int j= rand() % MAX_CITIES;
	while (i == j)
	{
		j= rand() % MAX_CITIES;
	}
	int temporal= tour[i];
	tour[i]= tour[j];
	tour[j]= temporal;
}

int main(int argc, char * argv[]){
	srand(time(NULL));
	MPI_Init(&argc, &argv);
	int rank, size;
	instancia data;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int numIteraciones= atoi(argv[i]);
	double matrizDistancia[MAX_CITIES][MAX_CITIES];
	double resultado=1e9; //darle un valor grande para que al final, no tome el 0 como minimo e imprima 0 o valores basura
	//vamos a generar una solucion "inicial"
	double resultado_vecino;
	int solucion[MAX_CITIES];
	srand(time(NULL)+rank);
	if (rank == 0)
	{
		calcularDistancia(matrizDistancia,data);//aqui nosotros tenemos la matriz hecha
		leerInstancia(&data);//la instancia son los datos del problema que queremos resolver (en el problema ustedes tienen archivos ".tsp")
	}

	MPI_Bcast(matrizDistancia,MAX_CITIES*MAX_CITIES, MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	if (rank != 0)
	{
		int vecino[MAX_CITIES];
		generaTour(solucion); // generar tour aleatorio pero es necesario tambien evaluar la solucion (funcion objetivo)
		for (int iter=0; iter<numIteraciones; iter++)
		{
			for (int i=0; i<MAX_CITIES; i++)
			{
				vecino[i]=solucion[i];
			}
			resultado= evaluar_tour(solucion,matrizDistancia);
			//printf("\nEl costo de la funcion objetivo es: %lf\n",resultado);
			resultado_vecino= evaluar_tour(solucion,matrizDistancia);
			//printf("\nEl costo de la funcion objetivo del vecino es: %lf\n",resultado_vecino);
			if (resultado_vecino < resultado)
			{
				for (int i=0; i<MAX_CITIES; i++)
				{
					solucion[i]= vecino[i];
					resultado= resultado_vecino;
				}
			}
		}
	}
	double mejor_costo;
	// generar un struct con 2 valores resultado y rank usando mpi_minloc
	// lo que hace minloc es trabajar con un par de valores donde el primero es el que se compara y se le asocia el valor del rank
	struct 
	{
		double valor;
		int rank;
	}local,global;
	local.valor= resultado;
	local.rank= rank;

	MPI_Reduce(&local, &global,1,MPI_DOUBLE_INT,MPI_MINLOC,0,MPI_COMM_WORLD);
	int mejorTour[MAX_CITIES];
	if (rank == global.rank)
	{
		MPI_Send(solucion, MAX_CITIES, MPI_INT, 0,0, MPI_COMM_WORLD);
	}
	
	if (rank == 0)
	{
		MPI_Recv(mejorTour, MAX_CITIES, MPI_INT, global.rank,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("\nEl mejor costo de todos los procesos es: %lf\n",global.valor);
		for (int i=0; i<MAX_CITIES; i++)
		{
			printf("%d-> ", mejorTour[i]);
		}
		printf("d\n",mejorTour[0]);
	}
	

	MPI_Finalize();
}