
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_CITIES 1000
#define POP_SIZE 100
#define GENERATIONS 500
#define MUTATION_RATE 0.1

typedef struct{
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
}TSPData;

typedef struct{
    int tour[MAX_CITIES];
    double fitness;
}Individual;

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

void shuffle(int *tour, int n){
    for(int i = 0; i < n; i++) tour[i] = i;
    for(int i = 0; i < n; i++){
        int j = rand() % n;
        int tmp = tour[i];
        tour[i] = tour[j];
        tour[j] = tmp;
    }
}

void mutate(int *tour, int n){
    if((double)rand()/RAND_MAX < MUTATION_RATE){
        int i = rand() % n;
        int j = rand() % n;
        int tmp = tour[i];
        tour[i] = tour[j];
        tour[j] = tmp;
    }
}

void crossover(int *p1, int *p2, int *child, int n){
    int start = rand() % n;
    int end = start + rand() % (n - start);
    for(int i = 0; i < n; i++) child[i] = -1;
    for(int i = start; i < end; i++) child[i] = p1[i];
    int idx = end;
    for(int i = 0; i < n; i++){
        int gene = p2[i];
        int exists = 0;
        for(int j = 0; j < n; j++){
            if(child[j] == gene){
                exists = 1;
                break;
            }
        }
        if(!exists){
            if(idx == n) idx = 0;
            child[idx++] = gene;
        }
    }
}

void genetic_algorithm(TSPData *data){
    Individual population[POP_SIZE];
    Individual new_population[POP_SIZE];
    for(int i = 0; i < POP_SIZE; i++){
        shuffle(population[i].tour, data->num_cities);
        population[i].fitness = total_distance(data, population[i].tour);
    }

    for(int gen = 0; gen < GENERATIONS; gen++){
        for(int i = 0; i < POP_SIZE; i++){
            int p1 = rand() % POP_SIZE;
            int p2 = rand() % POP_SIZE;
            crossover(population[p1].tour, population[p2].tour, new_population[i].tour, data->num_cities);
            mutate(new_population[i].tour, data->num_cities);
            new_population[i].fitness = total_distance(data, new_population[i].tour);
        }

        for(int i = 0; i < POP_SIZE; i++) population[i] = new_population[i];
    }

    int best_idx = 0;
    for(int i = 1; i < POP_SIZE; i++){
        if(population[i].fitness < population[best_idx].fitness)
            best_idx = i;
    }

  printf("Mejor tour encontrado (GA): %.2f\n", population[best_idx].fitness);

    printf("Tour final:\n");
    for(int i = 0; i < data->num_cities; i++)
        printf("%d->", population[best_idx].tour[i] + 1); //+1 para alinear con el índice del .tsp
    
    printf("%d\n", population[best_idx].tour[0] + 1); //vuelta al inicio

}

int main(){
    srand(time(NULL));
    TSPData data;
    FILE *fp = fopen("eil101.tsp", "r");
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

    genetic_algorithm(&data);
    return 0;
}
