import numpy as np
from scipy import interpolate

def descartacion_tramo_mejoras(datos, lastTramo):
    """
    Filtra tramos donde el IRI mejora más que la tolerancia del error de medicion.
    Args:
        datos: datos a analizar

    Returns: datos filtrados

    """
    etiquetas_tramos = np.unique(datos[:, 0])
    bad_indexes = []
    bad_indexes_ERROR = []
    
    for lbl in etiquetas_tramos:
        current_lbl_indexes = datos[:, 0] == lbl
        
        # Nos devuelve en True aquellas filas del tramo actual del "for"

        # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
        # la vamos a usar para eliminar registros si hubo mejoras
        indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
        indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

        # diff calcula la diferencia sobre el mismo eje
        # en este caso, sobre la columna 6 del tramo entero
        iri_diff = np.diff(datos[current_lbl_indexes, 6])

        any_high_values_ERROR = np.any((iri_diff <= -0.3) & (iri_diff > -0.6))
        # En "any_high_values" nos devuelve True/False de aquellos tramos que verifica la condicion anterior

        if any_high_values_ERROR:
            max_high_value_ERROR = np.max(np.argwhere((iri_diff <= -0.3) & (iri_diff > -0.6)) + 1)
            # max_high_value nos devuelve la maxima fila (por tramo)
            # donde verifica la diferencia

            datos[indice_minimo_tramo + max_high_value_ERROR, 6] = datos[indice_minimo_tramo + max_high_value_ERROR-1, 6]
            # En "bad_indexes_ERROR" tenemos las filas que se van a eliminar porque hubo ERROR de medicion:
                # en lugar de descartarla, copiamos el valor del iri anterior
                # porque sino, cuando hago la polinomica, tengo que "inventar" los otros valores del vector que acabamos de eliminar

        # DESCARTAMOS MEJORAS
        # En lugar de descartarla vamos a colocarlas como nuevo tramo (desde año 0)
        any_high_values = np.any(iri_diff <= -0.6)
        if any_high_values:
            max_high_value = np.max(np.argwhere(iri_diff <= -0.6) + 1)
            anioMejora = datos[indice_minimo_tramo + max_high_value, 1]
            anioMax = datos[indice_maximo_tramo, 1]
            anio = 0
            if anioMejora <= anioMax / 2:
                fila = indice_minimo_tramo + max_high_value
                while fila <= indice_maximo_tramo:
                    datos[fila, 1] = anio  #Corregimos año
                    fila += 1
                    anio += 1

                fila = indice_minimo_tramo
                while fila < indice_minimo_tramo + max_high_value:
                    datos[fila, 0] = lastTramo + 1  #Corregimos tramo
                    fila += 1
                    
            else:
                fila = indice_minimo_tramo
                while fila < indice_minimo_tramo + max_high_value:
                    datos[fila, 0] = lastTramo + 1  #Corregimos tramo
                    fila += 1
                
                fila = indice_minimo_tramo + max_high_value
                while fila <= indice_maximo_tramo:
                    datos[fila, 1] = anio  #Corregimos año
                    fila += 1
                    anio += 1
      
            lastTramo += 1

    return datos


def search_polilinea(datos):
    etiquetas_tramos = np.unique(datos[:, 0])
    
    for lbl in etiquetas_tramos:
       current_lbl_indexes = datos[:, 0] == lbl

       indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
       indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))
       largo_tramo = indice_maximo_tramo - indice_minimo_tramo + 1
       anioMin = datos[indice_minimo_tramo, 1]
       anioMax = datos[indice_maximo_tramo, 1]

       #Hacemos la polilinea si tenemos mas de dos puntos
       if anioMax > 1:  
           iri = [None] * largo_tramo
           
           # En "x" se guarda los años absolutos que quedaron sin los errores de medicion y sin las mejoras...
           # En "y" se guarda los iris correspondientes al "x" anterior...
           x = datos[indice_minimo_tramo:indice_maximo_tramo + 1, 1]
           y = datos[indice_minimo_tramo:indice_maximo_tramo + 1, 6]
           
           # poly crea la tupla (x,y) anterior
           poly = [tuple((a, b)) for a, b in zip(datos[indice_minimo_tramo:indice_maximo_tramo + 1, 1],
                                                 datos[indice_minimo_tramo:indice_maximo_tramo + 1, 6])]
           
           l = len(x)
           # linspace(start,stop,num,endpoint=True,retstep=False)
           t = np.linspace(0, 1, l-2, endpoint=False)   # Crea un array con valor inicial "start", valor final "stop" y "num" elementos
           t = np.append([0, 0, 0], t)
           t = np.append(t, [1, 1, 1])

           tck0 = [t, [x,y], 3]
           elem = (anioMax-anioMin)*10
           u3 = np.linspace(0, 1, elem, endpoint=False)
           out0 = interpolate.splev(u3, tck0)
           outArray = np.array(out0)

           # En outArray[0,i] figuran todos los puntos del eje x
           # En outArray[1,i] figuran todos los puntos del eje y
           index = indice_minimo_tramo
           num = 0
           while (index <= indice_maximo_tramo) and (num < int(elem)):
              # Recorremos outArray[0,num], entramos cuando encontramos el proximo mayor
              if outArray[0,num] > anioMin:
                 if anioMin-outArray[0,num-1] > outArray[0,num]-anioMin:
                    datos[index, 6] = outArray[1,num]
                 else:
                    datos[index, 6] = outArray[1,num-1]
                 anioMin += 1
                 index += 1
              num += 1
          
    return datos
	
		
def forzar_ascendente(datos, lastTramo):
    etiquetas_tramos = np.unique(datos[:, 0])

    # Vamos a "forzar" para que queden ascendente los iri
    #test_data = list()
    #train_lbl_draw = list()
    
    for lbl in etiquetas_tramos:
       current_lbl_indexes = datos[:, 0] == lbl

       indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
       indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))

       iri_diff = np.diff(datos[current_lbl_indexes, 6])
       any_high_values_ASC = np.any(iri_diff < 0)

       while any_high_values_ASC:
          high_value_ASC = (iri_diff < 0).nonzero()[0] + 1
          min_high_value = np.min(np.argwhere(iri_diff < 0))

          iriMin = datos[indice_minimo_tramo + min_high_value, 6]
          datos[indice_minimo_tramo + min_high_value + 1, 6] = iriMin

          #Volvemos a fijarnos si existe algun valor en negativo
          iri_diff = np.diff(datos[current_lbl_indexes, 6])
          any_high_values_ASC = np.any(iri_diff < 0)


       # En indice_maximo_tramo se guarda la ultima medicion de cada tramo
       # ... donde cada tmda = tmdaAnterior * 1.02
       # ... deflex = deflexAnterior
       
    return datos
