"""
Definicion formal Maquina de Turing:

Es una séptupla siendo M = (Q, Σ, Γ, δ, q0, B, F) donde:
- Q: conjunto finito de estados
- Σ: conjunto finito de simbolos de entrada
- Γ: conjunto de simbolos de la cinta (Σ ⊆ Γ)
- δ: funcion de transicion δ(q, X) = (p, Y, D)
- q0: estado inicial
- B: simbolo blanco (B ∈ Γ, B ∉ Σ)
- F: conjunto de estados de aceptacion
"""

def maquina_turing(cinta_inicial):
    # Configuracion inicial de la maquina
    # Convertimos la cadena de entrada en una lista para poder modificarla
    # Añadimos '_' al final como simbolo blanco (B en la definicion formal)
    cinta= list(cinta_inicial) + ['_']  # '_' representa el simbolo blanco B
    
    # Inicializamos el cabezal en la posicion 0 (primer simbolo de la cinta)
    cabezal= 0  # Posicion inicial del cabezal de lectura/escritura
    
    # Estado inicial de la maquina (q0 en la definicion formal)
    estado= 'q0'  # Estado actual de la maquina
    
    """
    Tabla de transiciones (δ):
    --> (estado_actual, simbolo_leido) ; (simbolo_a_escribir, movimiento, nuevo_estado) donde:

    - simbolo_a_escribir: reemplaza el simbolo leido en la cinta
    - movimiento: 'D' (derecha), 'I' (izquierda), 'P' (parar/detenerse)
    - nuevo_estado: estado al que transiciona la maquina
    """
    transiciones= {
        # Estado q0: estado inicial
        ('q0', '0'): ('_', 'D', 'q1'),  # Si lee '0', lo borra dejando "blanco", mueve derecha y va a q1
        ('q0', '1'): ('_', 'D', 'q2'),  # Si lee '1', lo borra dejando "blanco", mueve derecha y va a q2
        ('q0', '_'): ('_', 'P', 'aceptacion'),  # Si lee blanco, para, acepta (cadena vacia)
        


        # Estado q1: procesando '0's
        ('q1', '0'): ('0', 'D', 'q1'),  # Si lee '0', lo deja, mueve derecha y se queda en q1
        ('q1', '1'): ('1', 'D', 'q1'),  # Si lee '1', lo deja, mueve derecha y se queda en q1
        ('q1', '_'): ('_', 'I', 'q3'),  # Si lee blanco, lo deja, mueve izquierda (retrocede para verificar) y va a q3
        


        # Estado q2: procesando '1's
        ('q2', '0'): ('0', 'D', 'q2'),  # Si lee '0', lo deja, mueve derecha y se queda en q2
        ('q2', '1'): ('1', 'D', 'q2'),  # Si lee '1', lo deja, mueve derecha y se queda en q2
        ('q2', '_'): ('_', 'I', 'q4'),  # Si lee blanco, lo deja, mueve izquierda (retrocede para verificar) y va a q4
        


        # Estado q3: verificando que el ultimo simbolo sea '0'
        ('q3', '0'): ('_', 'I', 'q5'),  # Si lee '0', lo borra dejando "blanco", mueve izquierda y va a q5
        ('q3', '1'): ('1', 'P', 'dump'),  # Si lee '1', lo deja, para y se va a dump (rechaza)
        ('q3', '_'): ('_', 'P', 'aceptacion'),  # Si lee blanco, lo deja, para y acepta (el ultimo coincide con el primero que leyo '0')
        



        # Estado q4: verificando que el ultimo simbolo sea '1'
        ('q4', '1'): ('_', 'I', 'q5'),  # Si lee '1', lo borra dejando "blanco", mueve izquierda y va a q5
        ('q4', '0'): ('0', 'P', 'dump'),  # Si lee '0', lo deja, para y se va a dump (rechaza)
        ('q4', '_'): ('_', 'P', 'aceptacion'),  # Si lee blanco, lo deja, para y acepta (el ultimo coincide con el primero que leyo '1')
        



        # Estado q5: retrocediendo al inicio
        ('q5', '0'): ('0', 'I', 'q5'),  # Retrocede sobre '0's
        ('q5', '1'): ('1', 'I', 'q5'),  # Retrocede sobre '1's
        ('q5', '_'): ('_', 'D', 'q0'),  # Al llegar al inicio, reinicia el proceso
    }
    
    # Ciclo principal de ejecucion
    # La maquina sigue procesando hasta llegar a un estado de aceptacion o rechazo
    while estado not in ['aceptacion', 'dump']:
        # Lee el simbolo actual bajo el cabezal
        simbolo= cinta[cabezal]  # Obtiene el simbolo en la posicion actual del cabezal
        
        # Busca la transicion correspondiente al estado y simbolo actuales
        accion= transiciones.get((estado, simbolo), None)  # None si no hay transicion definida
        
        # Si no hay transicion definida se rechaza (por ejemplo el vacio) si se da un simbolo que no este en el alfabeto 0,1
        if not accion:
            estado= 'dump'  
            break
        
        # Desempaqueta la accion: que escribir, como moverse y nuevo estado
        nuevo_simbolo, movimiento, nuevo_estado = accion
        
        # Escribe el nuevo simbolo en la cinta
        cinta[cabezal]= nuevo_simbolo  # reemplaza el simbolo actual
        
        # Mueve el cabezal segun la instruccion
        if movimiento == 'D':
            cabezal+= 1  # mueve a la derecha
            # Si se sale de la cinta, añade un nuevo blanco (simula cinta infinita)
            if cabezal == len(cinta):
                cinta.append('_')
        elif movimiento == 'I':
            cabezal-= 1  # mueve a la izquierda
            # No permite que el cabezal sea negativo (cinta infinita solo a la derecha)
            if cabezal < 0:
                cabezal= 0
        
        # Actualiza el estado de la maquina
        estado= nuevo_estado
    
    # Regresa si la cadena fue aceptada y el estado final de la cinta (sin '_'   "blancos" al final)
    return estado == 'aceptacion', ''.join(cinta).strip('_')  # strip('_') elimina '_'    "blancos" al final antes de imprimir




cadena= "0110"



es_palindromo, cinta_final= maquina_turing(cadena)


if es_palindromo == True:
    print(f"\nLa cadena '{cadena}' es palindromo\n")
else: 
    print(f"\nLa cadena '{cadena}' no es palindromo\n")

#print(f"Cinta final: {cinta_final}")