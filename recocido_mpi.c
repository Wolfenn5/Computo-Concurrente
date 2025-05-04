/*
 * PROYECTO FINAL - IMPLEMENTACIÓN CON MPI
 * 
 * Implementación del algoritmo de Recocido Simulado para resolver el 
 * problema del agente viajero (TSP), utilizando N procesos. 
 * Cada proceso genera una solución independiente y, al finalizar, 
 * se selecciona la de menor costo entre todas.
 *
 * Compilación:
 *   $ mpicc -std=c99 tspmpi.c -o tspmpi -lm
 *  
 * Ejecución:
 *   $ mpiexec -n N --hostfile lista_nodos.txt ./tspmpi
 * 
 * Donde N es el número de procesos que se desean usar.
 * lista_nodos.txt contiene el nombre de todos los nodos que se van a utlizar 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#define MAX_CITIES 1024

typedef struct {
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
} TSPData;

double distance(double x1, double y1, double x2, double y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

double total_distance(TSPData *data, int *tour){
    double dist = 0.0;
    for(int i = 0; i < data->num_cities - 1; i++){
        dist += distance(data->x[tour[i]], data->y[tour[i]], data->x[tour[i+1]], data->y[tour[i+1]]);
    }
    dist += distance(data->x[tour[data->num_cities-1]], data->y[tour[data->num_cities-1]], data->x[tour[0]], data->y[tour[0]]);
    return dist;
}

void swap(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void shuffle(int *tour, int n){
    for(int i = 0; i < n; i++) tour[i] = i;
    for(int i = 0; i < n; i++){
        int j = rand() % n;
        swap(&tour[i], &tour[j]);
    }
}

void simulated_annealing(TSPData *data, int *local_tour){
    int current_tour[MAX_CITIES];
    shuffle(current_tour, data->num_cities);
    double temp = 1000.0;
    double alpha = 0.995;
    int max_iter = 100000;
    double current_cost = total_distance(data, current_tour);

    for(int i = 0; i < data->num_cities; i++) local_tour[i] = current_tour[i];

    for(int iter = 0; iter < max_iter; iter++){
        int i = rand() % data->num_cities;
        int j = rand() % data->num_cities;
        swap(&current_tour[i], &current_tour[j]);
        double new_cost = total_distance(data, current_tour);
        if(new_cost < current_cost || exp((current_cost - new_cost) / temp) > (double)rand()/RAND_MAX){
            current_cost = new_cost;
            for(int k = 0; k < data->num_cities; k++) local_tour[k] = current_tour[k];
        } else {
            swap(&current_tour[i], &current_tour[j]); // revert
        }
        temp *= alpha;
    }

    //printf("Costo final: %.2f\n", current_cost);
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);//Inizializacion del entorno MPI 
    double inicio_time = MPI_Wtime(); //Inicio del tiempo de ejecucion 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//Identificador del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size);//Numero total de procesoss
    srand(time(NULL) + rank); //Inicializacion de la semilla aleatoria, el valor de rank se agrega para genera una semilla diferente en cada proceso 

    TSPData data;


    if(rank==0){ //Solo el proceso raiz lee el archivo .tsp
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

    //El proceso raiz se encarga de compartir los datos con los demas procesos, esto es más eficiente a comparacion de que cada proceso abra y leea el archivo 
    //la funcion  de MPI_Bcast permite distribuir a todos los procesos 
    MPI_Bcast(&data.num_cities, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data.x, MAX_CITIES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(data.y, MAX_CITIES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_tour[MAX_CITIES];
    simulated_annealing(&data, local_tour); //Cada proceso genera una solucion
    double local_cost = total_distance(&data, local_tour); //Calculo del costo del tour local

    //Con una esturctura podemos almacenar los datos de cada proceso, 'encapsularlos' y poderlos enviar.
    struct {
        double cost;
        int rank;
    }local, global;
    
    local.cost = local_cost; //Asignacion del costo del proceso local
    local.rank = rank; //Asignacion del identificador del proceso local 

    //De todos los procesos realizados, se manda el de menor, con la funcion MPI_MINLOC se selecciona el menor
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    int best_tour[MAX_CITIES];

    if(rank == global.rank){ //Solo el proceso con el menor costo envia su tour al proceso raiz
        MPI_Send(local_tour, data.num_cities, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){ //Solo el proceso raiz recibe el mejor tour 
        MPI_Recv(best_tour, data.num_cities, MPI_INT, global.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double fin_time = MPI_Wtime(); //Final del tiempo de ejecucion
    if(rank == 0){ 
        printf("Mejor costo global: %.3f\n", global.cost);
        printf("Tour final:\n");
        for (int i = 0; i < data.num_cities; i++) {
            printf("%d->", best_tour[i] + 1);
        }
        printf("%d\n", best_tour[0] + 1);
        printf("Tiempo total: %.6f segundos\n", fin_time - inicio_time);
    }

    MPI_Finalize();//Finalizacion del entorno MPI 
    return 0;
}
