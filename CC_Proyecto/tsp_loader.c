#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CITIES 7397
#define MAX_NAME_LEN 64

typedef struct{
    char name[MAX_NAME_LEN];
    char file[MAX_NAME_LEN];
    int num_cities;
    double x[MAX_CITIES];
    double y[MAX_CITIES];
} TSPInstance;

void load_instance(TSPInstance *inst){
    FILE *fp = fopen(inst->file, "r");
    if(!fp){
        printf("No se pudo abrir el archivo: %s\n", inst->file);
        exit(1);
    }

    char line[256];
    while(fgets(line, sizeof(line), fp))
        if(strncmp(line, "NODE_COORD_SECTION", 18) == 0)
            break;
    

    int index;
    double x, y;
    for (int i = 0; i < inst->num_cities; i++){
        fscanf(fp, "%d %lf %lf", &index, &x, &y);
        inst->x[i] = x;
        inst->y[i] = y;
    }
    fclose(fp);
    printf("Instancia %s cargada con %d ciudades.\n", inst->name, inst->num_cities);
}

int main(){
    TSPInstance instances[3] = {
        {"berlin52", "berlin52.tsp", 52},
        {"eil101", "eil101.tsp", 101},
        {"pr1002", "pr1002.tsp", 1002}
    };

    printf("Selecciona una instancia:\n");
    for (int i = 0; i < 3; i++)
        printf("%d. %s\n", i + 1, instances[i].name);
    

    int choice;
    printf("Opción: ");
    scanf("%d", &choice);
    if(choice < 1 || choice > 3){
        printf("Opción inválida.\n");
        return 1;
    }

    TSPInstance selected = instances[choice - 1];
    load_instance(&selected);

    return 0;
}
