#include <stdio.h>
//#include <mpi.h>
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

int main(int argc, char * argv[]){
	srand(time(NULL));
	//MPI_Init(&argc, &argv);
	int rank, size;
	instancia data;
	//MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//MPI_Comm_size(MPI_COMM_WORLD,&size);
	leerInstancia(&data);//la instancia son los datos del problema que queremos resolver (en el problema ustedes tienen archivos ".tsp")

	double matrizDistancia[MAX_CITIES][MAX_CITIES];
	calcularDistancia(matrizDistancia,data);//aqui nosotros tenemos la matriz hecha
	//vamos a generar una solucion "inicial"
	int solucion[MAX_CITIES];
	generaTour(solucion);
	
//	MPI_Finalize();
}