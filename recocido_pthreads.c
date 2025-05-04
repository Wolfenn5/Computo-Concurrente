/*
 * PROYECTO FINAL. IMPLEMENTACION CON PTHREAD
 * Implementación del algoritmo de recocido simulado para resolver el 
 * problema del agente viajero (TSP) mediante la utilización de n hilos
 * de ejecución; los hilos se encargan de 1/n de las ejecuciones
 * del algoritmo, respectivamente.
 *
 * compilacion:
 *  $ gcc recocido_pthreads.c -o rec_pthreads -lm
 *  
 * Ejecucion:
 *  $ ./rec_pthreads <<archivo.tsp>> <<iteraciones>> <<numero hilos>>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <float.h>

#define MAX_CITIES 1500
// numero de hilos utilizados

int max_iter;
int num_HILOS;

// estructura para la instancia
typedef struct{
    double x[MAX_CITIES];
    double y[MAX_CITIES];
    int num_cities;
}TSPData;

// estructura para determinar el mejor costo entre hilos
typedef struct{
    TSPData* data;
    int* best_global_tour;
    double* best_global_cost;
    pthread_mutex_t* mutex; // mutex para espera
}ThreadData;

// calcula la distancia entre locaciones
double distance(double x1, double y1, double x2, double y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// calula distancia total del tour
double total_distance(TSPData *data, int *tour) {
    double dist = 0.0;
    for(int i=0; i<data->num_cities - 1; i++){
        dist += distance(data->x[tour[i]], data->y[tour[i]], 
                       data->x[tour[i+1]], data->y[tour[i+1]]);
    }
    dist += distance(data->x[tour[data->num_cities-1]], 
                    data->y[tour[data->num_cities-1]], 
                    data->x[tour[0]], data->y[tour[0]]);
    return dist;
}

// intercambio de locaciones
void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// para generar la solución de forma aleatoria
void shuffle(int *tour, int n) {
    for(int i = 0; i<n; i++) tour[i] = i;
    for(int i=0; i < n; i++){
        int j = rand() % n;
        swap(&tour[i], &tour[j]);
    }
}

// recocido simulado para cada hilo 
void* thread_simulated_annealing(void* arg) {
    ThreadData* tdata = (ThreadData*)arg;
    TSPData* data = tdata->data;
    
    int current_tour[MAX_CITIES];
    int best_local_tour[MAX_CITIES];
    // genera primera solución
    shuffle(current_tour, data->num_cities);
    double temp = 1000.0;
    double alpha = 0.895;
    // 1 millón de iteraciones totales divididas en 4 hilos
    // int max_iter = 250000;
    double current_cost = total_distance(data, current_tour);
    double best_local_cost = current_cost;
    // primer copia inicial, primer solcuión es la mejor al inicio
    memcpy(best_local_tour, current_tour, sizeof(int)*data->num_cities);

    for(int iter = 0; iter < max_iter; iter++) {
        int i = rand() % data->num_cities;
        int j = rand() % data->num_cities;
        swap(&current_tour[i], &current_tour[j]);
        double new_cost = total_distance(data, current_tour);
        
        //metropolis
        if(new_cost < current_cost || exp((current_cost - new_cost)/temp) > (double)rand()/RAND_MAX) {
            current_cost = new_cost;
            if(current_cost < best_local_cost) {
                best_local_cost = current_cost;
                // función para copiar el mejor tour encontrado hasta el momento
                // a la sección de memoria correspondiente con tamaño (int)[num_cities]
                memcpy(best_local_tour, current_tour, sizeof(int)*data->num_cities);
            }
        }else{
            // intercambbio
            swap(&current_tour[i], &current_tour[j]);
        }
        temp *= alpha;
    }

    // Actualizar el mejor global de manera segura
    pthread_mutex_lock(tdata->mutex);
    if(best_local_cost < *(tdata->best_global_cost)) {
        *(tdata->best_global_cost) = best_local_cost;
        // actualización de la mejor solucion (tour)
        memcpy(tdata->best_global_tour, best_local_tour, sizeof(int)*data->num_cities);
    }
    pthread_mutex_unlock(tdata->mutex);

    pthread_exit(NULL);
}

// función para crear los hilos y echar a andar el algoritmo de cristalización
void parallel_simulated_annealing(TSPData *data, int *best_tour){
    pthread_t threads[num_HILOS];
    ThreadData tdata[num_HILOS];
    pthread_mutex_t mutex;
    double best_global_cost = DBL_MAX;

    pthread_mutex_init(&mutex, NULL);

    //crea los hilos
    for(int i=0; i<num_HILOS; i++) {
        tdata[i].data = data;
        tdata[i].best_global_tour = best_tour;
        tdata[i].best_global_cost = &best_global_cost;
        tdata[i].mutex = &mutex;
        pthread_create(&threads[i], NULL, thread_simulated_annealing, &tdata[i]);
    }

    //esperar a que todos los hilos terminen
    for(int i=0; i<num_HILOS; i++){
        pthread_join(threads[i], NULL);
    }

    // destruye mutex de espera
    pthread_mutex_destroy(&mutex);
    // mejor solución encontrada
    printf("costo final: %.2f \n", best_global_cost);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    TSPData data;

    //Medicion de tiempo
    clock_t inicio, fin;
    double tiempo_cpu;
    
    // ingresamos archivo por  nombre
    if(argc != 4){
        printf("Use así: %s <<archivo.tsp>> <<iteraciones>> <<numero hilos>>\n", argv[0]);
        return 1;
    }
    
    num_HILOS = atoi(argv[3]);
    max_iter = atoi(argv[2])/num_HILOS;

    FILE *fp = fopen(argv[1], "r");
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
    // ejecuta el recocido simulado de dorma paralela :)
    // inicio de cuenta de tiempo
    inicio = clock(); 
    parallel_simulated_annealing(&data, best_tour);
    // final de conteo de tiempo
    fin = clock();
    tiempo_cpu = ((double)(fin - inicio)) / CLOCKS_PER_SEC;  // Tiempo en segundos
    printf("Tiempo de CPU: %f segundos\n", tiempo_cpu);
    /*
    printf("Tour final:\n"); //mejor solucion encontrada :)
    for (int i = 0; i < data.num_cities; i++)
        printf("%d->", best_tour[i] + 1);
    
    printf("%d\n", best_tour[0] + 1);
    */
    return 0;
}
