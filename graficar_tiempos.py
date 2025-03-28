import matplotlib.pyplot as plt

# tiempo promedio de ejecuciones
tiempo_secuencial= [60] # promedio de resultados de ejecucion del algoritmo secuencial


# promedio de resultados de ejecucion del algoritmo multihilado 
hilos_cpu= [1,2,4,8,16,32]
tiempos_cpu= [58,40,36,30,25,15] #poner ejecuciones reales
speedup_cpu= [tiempo_secuencial / tiempo for tiempo in tiempos_cpu]
print(speedup_cpu)


# promedio de resultados de ejecucion del algoritmo paralelo cuda
hilos_gpu= [64,128,256,512,1024]
tiempos_gpu=[12,9,7,6,5] #poner valores reales
speedup_gpu= [tiempo_secuencial / tiempo for tiempo in tiempos_gpu]
print(speedup_gpu)


plt.axhline(y=tiempo_secuencial, color= 'r', linestyle='--', label='secuencial') # esta se ocupa para la ejecucion secuencial

plt.plot(hilos_cpu, tiempos_cpu, marker='o', label= 'Multihilo CPU')
plt.plot(hilos_cpu, tiempos_cpu, marker='s', label= 'Multihilo GPU')


plt.title("Grafica de tiempos de ejecucion")
plt.xlabel('Numero de hilos o tamaño del bloque')
plt.ylabel('Tiempo de ejecucion promedio (s)')
plt.grid()
plt.legend()
plt.subplot(1,2,2) # de la grafica que se genero esta va a ser la segunda grafica
plt.plot(hilos_cpu, speedup_cpu, marker='o', label= 'Speedup CPU')
plt.plot(hilos_gpu, speedup_gpu, marker='s', label= 'Speedup GPU')
plt.title('Grafica del speedup')
plt.xlabel('Numero de hilos o tamaño del bloque')
plt.ylabel('speedup')
plt.show()