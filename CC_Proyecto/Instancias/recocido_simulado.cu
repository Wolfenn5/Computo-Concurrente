#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define MAX_CITIES 1000

typedef struct {
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
} TSPData;

__device__ double distance(double x1, double y1, double x2, double y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

__device__ double total_distance(TSPData *data, int *tour){
    double dist = 0.0;
    for(int i = 0; i < data->num_cities - 1; i++){
        dist += distance(data->x[tour[i]], data->y[tour[i]], data->x[tour[i+1]], data->y[tour[i+1]]);
    }
    dist += distance(data->x[tour[data->num_cities-1]], data->y[tour[data->num_cities-1]], data->x[tour[0]], data->y[tour[0]]);
    return dist;
}

__device__ void swap(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ void shuffle(int *tour, int n, curandState *state) {
    for(int i = 0; i < n; i++) tour[i] = i;
    for(int i = 0; i < n; i++) {
        int j = curand(state) % n;
        int tmp = tour[i];
        tour[i] = tour[j];
        tour[j] = tmp;
    }
}




__global__ void kernel_simulated_annealing(TSPData *data, int *best_tours, double *best_costs, int max_iter, unsigned long seed) 
{
    int idHilo = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Inicializar el estado del generador curand (para aleatoiros)
    curandState state;
    curand_init(seed, idHilo, 0, &state); // semilla, idHilo es el id unico de hilo


    // Cada hilo tendrá su propio tour y estado (semilla para aleatorios)
    int current_tour[MAX_CITIES];
    shuffle(current_tour, data->num_cities, &state);
    double temp = 1000.0;
    double alpha = 0.995;
    double current_cost = total_distance(data, current_tour);
    

    // Copiar a mejor tour inicial
    for(int i = 0; i < data->num_cities; i++) 
    {
        best_tours[idHilo * data->num_cities + i] = current_tour[i]; // en vez de best_tour[i], se calcula el indice a partir del id del hilo
    }
    best_costs[idHilo] = current_cost; // best cost se usa para mandar el toru con menor costo despues al host
    

    // Algoritmo de simulated annealing
    for(int iter = 0; iter < max_iter; iter++) 
    {
        // Generar indices aleatorios i y j usando curand
        int i = curand(&state) % data->num_cities;
        int j = curand(&state) % data->num_cities;
        swap(&current_tour[i], &current_tour[j]);
        
        double new_cost = total_distance(data, current_tour);
        
        // Conocer cual es el de mejor costo a partir de los costos almacenados en el arreglo bes_cost
        if(new_cost < current_cost || exp((current_cost - new_cost) / temp) > curand_uniform(&state))
        {
            current_cost = new_cost;
            for(int k = 0; k < data->num_cities; k++) 
            {
                best_tours[idHilo * data->num_cities + k] = current_tour[k]; // en vez de best_tour[k], se calcula el indice a partir del id del hilo
            }
            best_costs[idHilo] = current_cost;
        } 
        else 
        {
            swap(&current_tour[i], &current_tour[j]); // revert
        }
        temp *= alpha;
    }
}



void simulated_annealing(TSPData *data, int *best_tour) 
{
    // Calcular el numero de hilos a ocupar
    cudaDeviceProp propiedades; // sacar propiedades del dispositivo
    cudaGetDeviceProperties (&propiedades,0);
    int num_threads= propiedades.maxThreadsPerBlock; // calcular tamaño optimo del bloque
    int num_blocks= (MAX_CITIES+num_threads-1) / num_threads; // calcular el numero de bloques ; N es el numero de datos (en este caso el tamaño de las ciudades), la formula es universal
    // Si se quiere saber el numero de hilos maximo se multiplica tam_bloque*numero_bloques

    
    // Reservar memoria en el dispositivo
    TSPData *d_data;
    int *d_best_tours;
    double *d_best_costs;
    int max_iter = 100000;
    
    cudaMalloc(&d_data, sizeof(TSPData));
    cudaMalloc(&d_best_tours, num_threads * data->num_cities * sizeof(int));
    cudaMalloc(&d_best_costs, num_threads * sizeof(double));
    
    // Copiar datos del host al dispositivo
    cudaMemcpy(d_data, data, sizeof(TSPData), cudaMemcpyHostToDevice);
    
    // Lanzar kernel
    unsigned long seed= time(NULL); // semilla utilizando la hora para utilizar con curand 
    kernel_simulated_annealing<<<num_blocks, num_threads>>>(d_data, d_best_tours, d_best_costs, max_iter, seed);
    
    // Copiar datos del dispositivo al host
    int *host_best_tours = (int*)malloc(num_threads * data->num_cities * sizeof(int));
    double *host_best_costs = (double*)malloc(num_threads * sizeof(double));
    cudaMemcpy(host_best_tours, d_best_tours, num_threads * data->num_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_best_costs, d_best_costs, num_threads * sizeof(double), cudaMemcpyDeviceToHost);
    

    // Encontrar el mejor tour entre todos los hilos (ya en el host)
    double min_cost= host_best_costs[0]; // se dira que el primer valor es el mejor (de inicio)
    int best_thread= 0; // se dira que el hilo que tuvo el mejor tour sera el 0 (de inicio)
    for(int i=1; i<num_threads; i++) 
    {
        if(host_best_costs[i] < min_cost) // si el elemento actual es menor al de menor costo (previo)
        {
            min_cost= host_best_costs[i];
            best_thread= i; // se obtiene que hilo fue el de menor costo para poder imprimir su tour despues
        }
    }
    // Copiar el mejor tour al resultado
    for(int i=0; i<data->num_cities; i++) 
    {
        best_tour[i]= host_best_tours[best_thread * data->num_cities + i]; // de todos los tours se obtiene el mejor dependiendo de que hilo fue 
    }
    
    printf("Costo final: %.2f\n", min_cost);

    // Liberar memoria 
    // Host
    free(host_best_tours);
    free(host_best_costs);
    // Dispositivo
    cudaFree(d_data);
    cudaFree(d_best_tours);
    cudaFree(d_best_costs);
}

int main() 
{
    srand(time(NULL));
    TSPData data;
    FILE *fp = fopen("berlin52.tsp", "r");
    if(!fp) {
        printf("No se pudo abrir el archivo.\n");
        return 1;
    }

    char line[128];
    while(fgets(line, sizeof(line), fp)) {
        if(strncmp(line, "NODE_COORD_SECTION", 18) == 0)
            break;
    }

    int index;
    double x, y;
    data.num_cities = 0;
    while(fscanf(fp, "%d %lf %lf", &index, &x, &y) == 3) {
        data.x[data.num_cities] = x;
        data.y[data.num_cities] = y;
        data.num_cities++;
    }
    fclose(fp);

    int best_tour[MAX_CITIES];
    simulated_annealing(&data, best_tour);

    printf("Tour final:\n");
    for (int i = 0; i < data.num_cities; i++)
        printf("%d->", best_tour[i] + 1); //+1 para coincidir con los índices de TSPLIB
    
    printf("%d\n", best_tour[0] + 1); //regresamos al inicio

    return 0;
}