import matplotlib.pyplot as plt


"""
Con tamaño de  datos de: 1,000 es decir 1,000 * 1,000 = 1,000,000
Tiempos del secuencial (parcial_multiplicacion_matrices_secuencial.c):
6.007795
6.531358
7.023011
6.745856
6.351849


Tiempos del paralelo, la GPU si acepta 1024 hilos por bloque (parcial_multiplicacion_matrices_paralelo.cu):
64 hilos:            128 hilos:          256 hilos:          512 hilos:          1024 hilos:
   0.011323            0.006662            0.017348            0.060456            0.207889
   0.011317            0.006903            0.017376            0.060214            0.211880
   0.011438            0.006915            0.017352            0.060541            0.201149
   0.011374            0.006863            0.017367            0.060423            0.200600
   0.011374            0.006858            0.017349            0.060653            0.200199

"""


# tiempo promedio de ejecuciones
tiempo_secuencial= 6.5319738 # promedio de resultados de ejecucion del algoritmo secuencial


# promedio de resultados de ejecucion del algoritmo secuencial ejecutado 5 veces
hilos_cpu= [1,1,1,1,1] # mas que hilos, son las veces que se ejecuto secuencialmente (en esencia es como si fuera un solo hilo)
tiempos_cpu= [6.007795,6.531358,7.023011,6.745856,6.351849] # promedio de los tiempos
speedup_cpu= [tiempo_secuencial / tiempo for tiempo in tiempos_cpu]
print(speedup_cpu)


# promedio de resultados de ejecucion del algoritmo paralelo cuda
hilos_gpu= [64,128,256,512,1024]
tiempos_gpu=[0.0113652,0.0068402,0.0173584,0.0604574,0.2043434] # promedio de los tiempos
speedup_gpu= [tiempo_secuencial / tiempo for tiempo in tiempos_gpu]
print(speedup_gpu)

plt.subplot(1,2,1) # 1 fila 2 columnas
plt.axhline(y=tiempo_secuencial, color= 'r', linestyle='--', label='secuencial') # esta se ocupa para la ejecucion secuencial
plt.plot(hilos_cpu, tiempos_cpu, marker='o', label= 'Multihilo CPU')
plt.plot(hilos_gpu, tiempos_gpu, marker='s', label= 'Multihilo GPU')
plt.title("Tiempos de ejecucion (multiplicacion de matrices")
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