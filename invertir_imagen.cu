#include <stdio.h>
#include <opencv2/opencv.hpp>

/* este programa hace la inversion de imagen*/
// como compilar: (va a dar varios warnings, es normal)
// nvcc invertir_imagen.cu -o invertir_imagen `pkg-config --cflags --libs opencv4` -std=c++11


__global__ void invertir_colores (unsigned char * imagen, int ancho, int alto, int canales)
{
    // calcular la posicion del hilo en x y y (la posicion del pixel)
    int x= blockIdx.x * blockDim.x + threadIdx.x; // columna del hilo en el bloque
    int y= blockIdx.y * blockDim.y + threadIdx.y; // fila del hilo en el bloque

    if (x<ancho && y<alto)
    {
        // es necesario linealizar la imagen  (arreglo de pixeles)
        // porquel as imagenes se procesan como arreglos de pixeles
        int id_x= (y*ancho+x)*canales; // posicion del pixel
        // se necesita invertir
        for (int i=0; i<canales; i++)
        {
            imagen[id_x+i]= 255-imagen[id_x+i];
        }
        
    }
    
}



int main(int argc, char const *argv[])
{
    //cargar la imagen y transformar a una matriz
    cv::Mat imagen= cv::imread("imagen.png",cv::IMREAD_COLOR);
    int ancho= imagen.cols; // numero de columnas
    int alto= imagen.rows; // numero de filas
    printf("\nAncho %d y alto %d",ancho, alto);
    int canales=imagen.channels();
    //si tenemos 3 canales es para el RGB
	//Ya que tenemos la info de la imagen, podemos calcular el tamaño de la misma
	//debemos recordar que size_t es un tipo de dato que se usa para representar el tamaño de la memoria
    // ya que se tiene la informacion de la imagen


    size_t tamanio= ancho*alto*canales*sizeof(unsigned char);
    // los pixeles se dan en RGB que son valores numericos entre 0 y 255


    // usando la GPU
    cudaDeviceProp propiedades;
    cudaGetDeviceProperties(&propiedades,0);
    printf("\nEl numero maximo de hilos por bloque es: %d", propiedades.maxThreadsPerBlock);
    printf("\nEl numero maximo de hilos por dimension x : %d, y: %d, z: %d", propiedades.maxThreadsDim[0], propiedades.maxThreadsDim[1], propiedades.maxThreadsDim[2]);


    // Ajustar el numero de hilos a ejecutar, por lo tanto, si el numero de hilos en x y el numero de hilos en y supera el limite de hilos por bloque, necesitamos reducir la dimension
    // sqrt(1024)= 32
    // 32*32 hilos


    int hilos_x= propiedades.maxThreadsDim[0];
    int hilos_y= propiedades.maxThreadsDim[1];

    //si sabemos que es cuadrada la GPU, sacamos directamente la raiz
    while (hilos_x*hilos_y >= propiedades.maxThreadsPerBlock) 
    {
        if (hilos_x >= hilos_y) // ir reduciendo la dimension con mayor valor
        {
            hilos_x--;
        }
        if (hilos_y >= hilos_x)
        {
            hilos_y--;
        }
    }
    
    printf("\nEl numero de hilos en x %d y en y %d", hilos_x, hilos_y);
    dim3 hilosBloque(hilos_x, hilos_y); //32*32*1= 1024
    dim3 malla((ancho + hilosBloque.x -1) / hilosBloque.x, (alto + hilosBloque.y -1) / hilosBloque.y); 
    // (1024+32-1) / 32= 32.9= 32 
    // malla(32*32)


    printf("\nHilos en el bloque: %d y %d", hilos_x, hilos_y);
    printf("\nBloques en la grid: %d y %d", malla.x, malla.y);


    //reservamos memoria en el host
	//ya lse hizo mas arriba


    // Reservar memoria en el dispositivo
    unsigned char * imagen_dispositivo;
    cudaMalloc((void**) &imagen_dispositivo, tamanio);

    // Copiar del host al dispositivo
    cudaMemcpy(imagen_dispositivo, imagen.data, tamanio, cudaMemcpyHostToDevice);

    // Lanzar el kernel
    invertir_colores<<<malla, hilosBloque>>>(imagen_dispositivo, ancho, alto, canales);

    // Join de cuda para esperar a los hilos
    cudaDeviceSynchronize();

    // Copiar del dispositivo al host
    cudaMemcpy(imagen.data, imagen_dispositivo, tamanio, cudaMemcpyDeviceToHost);


    cv::imwrite("imagen_invertida.png",imagen); // sacar la imagen invertida

    return 0;
}
