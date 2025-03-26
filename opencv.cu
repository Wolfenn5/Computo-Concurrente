#include <stdio.h>
#include <opencv2/opencv.hpp>

// este programa hace la inversion de imagen


__global__ invertir_colores (unsigned char * imagen, int ancho, int alto, int canales)
{
    // calcular la posicion del hilo en x y y (la posicion del pixel)
    int x= blockIdx.x * blockDim.x + threadIdx.x; // columna del hilo en el bloque
    int y= lockIdx.y * blockDim.y + threadIdx.y; // fila del hilo en el bloque

    if (x<ancho && y<alto)
    {
        // es necesario linealizar la imagen  (arreglo de pixeles)
        int id_x= (y*ancho+x)*canales; // posicion del pixel
        // se necesita invertir
        for (int i=0; i<canales; i++)
        {
            imagen[id_x+i]= 255-imagen[id_x+i]
        }
        
    }
    
}



int main(int argc, char const *argv[])
{
    //cargar la imagen y transformar a una matriz
    cv::Mat imagen= cv::imread("imagen.png",cv::IMREAD_COLOR);
    int ancho= imagen.cols;
    int alto= imagen.rows;
    int canales=imagen.channels();
    // ya que se tiene la informacion de la imagen


    size_t tamanio= ancho*alto*canales*sizeof(unsigned char);
    // los pixeles se dan en RGB que son valores numericos entre 0 y 255


    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades,0);
    printf("\nEl numero maximo de hilos por bloque es: %d", propiedades.maxThreadsPerBlock);
    printf("\nEl numero maximo de hilos por dimension x : %d, y: %d, z: %d", propiedades.maxThreadsDim[0], propiedades.maxThreadsDim[1], propiedades.maxThreadsDim[2]));


    // Ajustar el numero de hilos a ejecutar
    // sqrt(1024)= 32
    // 32*32 hilos


    int hilos_x= propiedades.maxThreadsDim[0];
    int hilos_y= propiedades.maxThreadsDim[1];

    while (hilos_x*hilos_y >= propiedades.maxThreadsPerBlock) // ir reduciendo la dimension con mayor valor
    {
        if (hilos_x >= hilos_y)
        {
            hilos_x--;
        }
        else
        {
            hilos_y--;
        }
    }
    
    printf("\nEl numero de hilos en x %d y en y %d", hilos_x, hilos_y);
    dim3 hilosBloque(hilos_x, hilos_y); //32*32*1
    dim3 malla((ancho + hilosBloque.x -1) / hilosBloque.x, (alto + hilosBloque.y -1) / hilosBloque.y); 
    // (1024+32-1) / 32= 32.9= 32 
    // malla(32*32)
    printf("\nHilos en el bloque: %d y %d", hilos_x, hilos_y);
    printf("\nBloques en la grid: %d y %d", malla.x, malla.y);

    // Reservar memoria en el dispositivo
    unsigned char * imagen_dispositivo;
    cudaMalloc((void**) &imagen_dispositivo, tamanio);

    // Copiar del host al dispositivo
    cudaMemcpy(imagen_dispositivo, imagen.data, tamanio, cudaMemcpyHostToDevice);

    // Lanzar el kernel
    invertir_colores<<<malla, hilosBloque>>>(imagen_dispositivo, ancho, alto, canales);

    cudaDeviceSynchronize();

    // Copiar del dispositivo al host
    cudaMemcpy(imagen.data, imagen_dispositivo, tamanio, cudaMemcpyDeviceToDevice);


    cv::imwrite("imagen_invertida.png",imagen);

    return 0;
}
