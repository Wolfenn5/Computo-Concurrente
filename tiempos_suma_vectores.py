import matplotlib.pyplot as plt


"""
Con tamaño de  datos de: 1,000,000,000 
Tiempos del secuencial (suma_vectores_secuencial.c):
11.715476
7.319415
6.329879
6.664610
6.613568


Tiempos del paralelo (suma_vectores_MIMD.c):
 1 hilo:             2 hilos:            4 hilos:            8 hilos:            16 hilos:            32 hilos:
   6.685607            6.404634            6.369546            6.367297            6.373609
   6.332991            6.412074            6.364882            6.450158            6.427674
   6.389341            6.646051            6.360936            6.808293            6.711997
   6.796591            6.397805            6.530588            6.445955            9.135444
   6.420951            6.458792            6.330388            9.315028            6.315870


Tiempos del paralelo, la GPU si acepta 1024 hilos por bloque (suma_vectores_cuda.cu):
64 hilos:            128 hilos:          256 hilos:          512 hilos:          1024 hilos:
   0.007252            0.007216            0.007217            0.007239            0.006812
   0.007330            0.007388            0.007309            0.007419            0.006800
   0.007313            0.007519            0.007362            0.007412            0.006805
   0.007457            0.007446            0.007396            0.007445            0.006806
   0.007238            0.007402            0.007389            0.007367            0.006841

"""
# tiempo promedio de ejecuciones
tiempo_secuencial= 7.7285896 # promedio de resultados de ejecucion del algoritmo secuencial


# promedio de resultados de ejecucion del algoritmo multihilado 
hilos_cpu= [1,2,4,8,16,32]
tiempos_cpu= [6.5250962,6.4768205,6.391268,7.0773462,6.9929188, 6.9929188] #poner ejecuciones reales
speedup_cpu= [tiempo_secuencial / tiempo for tiempo in tiempos_cpu]
print(speedup_cpu)


# promedio de resultados de ejecucion del algoritmo paralelo cuda
hilos_gpu= [64,128,256,512,1024]
tiempos_gpu=[0.007318,0.0073942,0.0073346,0.0073764,0.0068128] #poner valores reales
speedup_gpu= [tiempo_secuencial / tiempo for tiempo in tiempos_gpu]
print(speedup_gpu)

plt.subplot(1,2,1) # 1 fila 2 columnas
plt.axhline(y=tiempo_secuencial, color= 'r', linestyle='--', label='secuencial') # esta se ocupa para la ejecucion secuencial
plt.plot(hilos_cpu, tiempos_cpu, marker='o', label= 'Multihilo CPU')
plt.plot(hilos_gpu, tiempos_gpu, marker='s', label= 'Multihilo GPU')
plt.title("Tiempos de ejecucion (suma de vectores")
plt.xlabel('Numero de hilos o tamaño del bloque')
plt.ylabel('Tiempo de ejecucion promedio (s)')
plt.grid()
plt.legend()


plt.subplot(1,2,2) # 1 fila 2 columnas, 2da grafica 
plt.plot(hilos_cpu, speedup_cpu, marker='o', label= 'Speedup CPU')
plt.plot(hilos_gpu, speedup_gpu, marker='s', label= 'Speedup GPU')
plt.title('Grafica del speedup')
plt.xlabel('Numero de hilos o tamaño del bloque')
plt.ylabel('Speedup')
plt.grid()
plt.legend()
plt.show()