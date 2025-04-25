
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_CITIES 1002
#define NUM_ANTS 50
#define MAX_ITER 200
#define ALPHA 1.0
#define BETA 5.0
#define RHO 0.5
#define QVAL 100.0

typedef struct {
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
} TSPData;

double dist_matrix[MAX_CITIES][MAX_CITIES];
double pheromone[MAX_CITIES][MAX_CITIES];

void compute_distances(TSPData *data){
    for(int i = 0; i < data->num_cities; i++){
        for(int j = 0; j < data->num_cities; j++){
            double dx = data->x[i] - data->x[j];
            double dy = data->y[i] - data->y[j];
            dist_matrix[i][j] = sqrt(dx*dx + dy*dy);
            pheromone[i][j] = 1.0;
        }
    }
}

double total_distance(int *tour, int n){
    double d = 0.0;
    for(int i = 0; i < n - 1; i++){
        d += dist_matrix[tour[i]][tour[i+1]];
    }
    d += dist_matrix[tour[n-1]][tour[0]];
    return d;
}

int select_next(int current, int *visited, int n){
    double prob[MAX_CITIES];
    double sum = 0.0;
    for(int i = 0; i < n; i++){
        if(!visited[i]){
            prob[i] = pow(pheromone[current][i], ALPHA) * pow(1.0 / dist_matrix[current][i], BETA);
            sum += prob[i];
        } else {
            prob[i] = 0.0;
        }
    }

    double r = ((double)rand() / RAND_MAX) * sum;
    double total = 0.0;
    for(int i = 0; i < n; i++){
        total += prob[i];
        if(total >= r)
            return i;
    }

    for(int i = 0; i < n; i++){
        if(!visited[i])
            return i;
    }
    return 0;
}

void ant_system(TSPData *data){
    int best_tour[MAX_CITIES];
    double best_len = 1e9;

    compute_distances(data);

    for(int iter = 0; iter < MAX_ITER; iter++){
        int ants[NUM_ANTS][MAX_CITIES];
        for(int k = 0; k < NUM_ANTS; k++){
            int visited[MAX_CITIES] = {0};
            ants[k][0] = rand() % data->num_cities;
            visited[ants[k][0]] = 1;
            for(int i = 1; i < data->num_cities; i++){
                ants[k][i] = select_next(ants[k][i-1], visited, data->num_cities);
                visited[ants[k][i]] = 1;
            }
        }

        // evaporación
        for(int i = 0; i < data->num_cities; i++){
            for(int j = 0; j < data->num_cities; j++){
                pheromone[i][j] *= (1.0 - RHO);
            }
        }

        for(int k = 0; k < NUM_ANTS; k++){
            double d = total_distance(ants[k], data->num_cities);
            for(int i = 0; i < data->num_cities - 1; i++){
                int from = ants[k][i];
                int to = ants[k][i+1];
                pheromone[from][to] += QVAL / d;
                pheromone[to][from] += QVAL / d;
            }
            pheromone[ants[k][data->num_cities-1]][ants[k][0]] += QVAL / d;
            pheromone[ants[k][0]][ants[k][data->num_cities-1]] += QVAL / d;

            if(d < best_len){
                best_len = d;
                for(int i = 0; i < data->num_cities; i++)
                    best_tour[i] = ants[k][i];
            }
        }
    }
    printf("Mejor tour encontrado (Ant System): %.2f\n", best_len);

    printf("Tour final:\n");

    for (int i = 0; i < data->num_cities; i++)
        printf("%d->", best_tour[i] + 1); // +1 si los índices en el archivo .tsp empiezan desde 1
    printf("%d\n", best_tour[0] + 1); // vuelta al inicio

}

int main(){
    srand(time(NULL));
    TSPData data;
    FILE *fp = fopen("pr1002.tsp", "r");
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

    ant_system(&data);
    return 0;
}
