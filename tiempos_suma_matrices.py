import matplotlib.pyplot as plt


# tiempo promedio de ejecuciones
tiempo_secuencial= 0.431646 # promedio de resultados de ejecucion del algoritmo secuencial


# promedio de resultados de ejecucion del algoritmo multihilado 
hilos_cpu= [2,4,8,16,32]
tiempos_cpu= [16.0589,16.9749,22.1053,27.7412,27.2112] # promedio de los tiempos
speedup_cpu= [tiempo_secuencial / tiempo for tiempo in tiempos_cpu]
eficiencia_cpu = [speedup / hilos for speedup, hilos in zip(speedup_cpu, hilos_cpu)]
P_cpu = [(hilos * (speedup - 1)) / (speedup * (hilos - 1)) if hilos > 1 else 0 
         for hilos, speedup in zip(hilos_cpu, speedup_cpu)]
print("SpeedUp CPU: ",speedup_cpu)
print("Eficiencia CPU: ", eficiencia_cpu)
print("Fraccion paralelizable estimada (P) CPU: ", P_cpu)


# promedio de resultados de ejecucion del algoritmo paralelo cuda
hilos_gpu= [64,128,256,512,1024] 
tiempos_gpu=[0.146221,0.144926,0.150378,0.148841,0.144171] # promedio de los tiempos
speedup_gpu= [tiempo_secuencial / tiempo for tiempo in tiempos_gpu]
eficiencia_gpu = [speedup / hilos for speedup, hilos in zip(speedup_gpu, hilos_gpu)]
P_gpu = [(hilos * (speedup - 1)) / (speedup * (hilos - 1)) if hilos > 1 else 0 
         for hilos, speedup in zip(hilos_gpu, speedup_gpu)]
print("\nSpeedUp GPU: ", speedup_gpu)
print("Eficiencia GPU: ", eficiencia_gpu)
print("Fraccion paralelizable estimada (P) GPU: ", P_gpu)

plt.subplot(1,2,1) # 1 fila 2 columnas
plt.axhline(y=tiempo_secuencial, color= 'r', linestyle='--', label='secuencial') # esta se ocupa para la ejecucion secuencial
plt.plot(hilos_cpu, tiempos_cpu, marker='o', label= 'Multihilo CPU')
plt.plot(hilos_gpu, tiempos_gpu, marker='s', label= 'Multihilo GPU')
plt.title("Tiempos de ejecucion (suma de matrices)")
plt.xlabel('Numero de hilos o numero de bloque')
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