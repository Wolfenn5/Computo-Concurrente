#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define MAX_CITIES 1000

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

void simulated_annealing(TSPData *data, int *best_tour){
    int current_tour[MAX_CITIES];
    shuffle(current_tour, data->num_cities);
    double temp = 1000.0;
    double alpha = 0.995;
    int max_iter = 100000;
    double current_cost = total_distance(data, current_tour);

    for(int i = 0; i < data->num_cities; i++) best_tour[i] = current_tour[i];

    for(int iter = 0; iter < max_iter; iter++){
        int i = rand() % data->num_cities;
        int j = rand() % data->num_cities;
        swap(&current_tour[i], &current_tour[j]);
        double new_cost = total_distance(data, current_tour);
        if(new_cost < current_cost || exp((current_cost - new_cost) / temp) > (double)rand()/RAND_MAX){
            current_cost = new_cost;
            for(int k = 0; k < data->num_cities; k++) best_tour[k] = current_tour[k];
        } else {
            swap(&current_tour[i], &current_tour[j]); // revert
        }
        temp *= alpha;
    }

    printf("Costo final: %.2f\n", current_cost);
}

int main(){
    srand(time(NULL));
    TSPData data;
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

    int best_tour[MAX_CITIES];
    clock_t inicio = clock();
    simulated_annealing(&data, best_tour);
    clock_t fin = clock();
    double tiempo_segundos = (double)(fin - inicio) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecucion: %.4f segundos\n", tiempo_segundos);

    printf("Tour final:\n");
    for (int i = 0; i < data.num_cities; i++)
        printf("%d->", best_tour[i] + 1); //+1 para coincidir con los índices de TSPLIB
    
    printf("%d\n", best_tour[0] + 1); //regresamos al inicio

    return 0;
}
