# Predecimos los ultimos 3 años conocidos
# + 3 años futuros

import os
import sys
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from sklearn import preprocessing
import plotearContinuo_RFR
import matplotlib
import funciones
   
def main(argv):
    """
    TransformApp entry point
    argv: command-line arguments (excluding this script's name)
    """

    '''
    El módulo "argparse" facilita la escritura de interfaces de línea de comandos fáciles de usar.
    El programa define qué argumentos requiere, y "argparse" descubrirá cómo analizar los que están fuera de sys.argv.
    El módulo "argparse" también genera automáticamente mensajes de ayuda y uso y emite errores cuando los usuarios dan al programa argumentos inválidos.
    '''
    parser = argparse.ArgumentParser(description="Script de modelado de evolucion de IRI")

    # Named arguments
    parser.add_argument("--verbose", "-v", help="Genera una salida detallada en consola", action="store_true")
    parser.add_argument("datos", help="Ruta al archivo de datos csv", metavar="DATA_PATH", nargs="?")
    parser.add_argument("--plot", "-p", help="Grafica las predicciones", action="store_true")

    args = parser.parse_args(argv)

    if not args.datos:
        print("Faltan argumentos posicionales")
        parser.print_usage()
        return 1

    if os.path.isfile(args.datos):
        data = np.loadtxt(args.datos, delimiter=",")
    else:
        # Cargo datos
        data = np.loadtxt('datos.csv', delimiter=",")

    # Nos quedamos con el nro. de tramo
    etiquetas_tramos=np.unique(data[:, 0])
    lastTramo = np.max(etiquetas_tramos)

    # Reemplazamos el año absoluto, por el relativo
    # Buscamos el menor año de evaluacion, restandoselo al año
    data[:, 1] -= np.min(data[:, 1])
     
    ####################################
    
    # descarto tramo cuando hay mejoras
    dataOriginal = data.copy()
    data = funciones.descartacion_tramo_mejoras(data, lastTramo)
    
	
     # En data tenemos los datos sin los errores de medicion, y sin las mejoras
    ####################################
    
    data = funciones.search_polilinea(data)
    # Ahora en data tenemos en la columna del IRI los iri corregidos
    # LOS AÑOS ANTERIORES A LA MEJORA LOS COLOCA COMO NUEVO TRAMO, DEJANDO COMO LOS ORIGINALES LOS QUE CORRESPONDEN SOLO QUE LOS INICIALIZA EN EL AÑO 0
	
    data = funciones.forzar_ascendente(data, lastTramo)
	
    data_lbl = data[:,-1].copy()
    data[:, -1] = np.roll(data[:, -1], 1)
	
    dataOriginal_lbl = dataOriginal[:,-1].copy()
    dataOriginal[:, -1] = np.roll(dataOriginal[:, -1], 1)
	
    for tramoTrain in np.unique(data[:, 0]):  
        current_lbl_indexes = data[:, 0] == tramoTrain
        indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
        indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))
        anioMax = data[indice_maximo_tramo, 1]
        
        #Hacemos la polilinea si tenemos mas de dos puntos
        if anioMax > 1:           
            # itero por los casos de corte para arreglar el iri
            case = indice_minimo_tramo
            m = data[case + 2, -1] - data[case + 1, -1]
            h = data[case + 1, -1]
            # evaluo en la recta en el x anterior (-1)
            jj = -1 * m + h - 0.5
            data[case, -1] = jj   # metemos ruido gaussiano sigma 0.5

        # Tenemos que agregarle al conjunto de entrenamiento todos los años menos los ultimos 3 que son los que queremos testear
        if anioMax >= 2:
            train_indexes = data[:, 0] != tramoTrain
            train_data_lbl = data_lbl[train_indexes].copy()
            train_data = data[train_indexes, 1:]
			
            train_lbl_draw = list()
            yPredict_RFR_draw = list()
			
            train_data_lbl = list(train_data_lbl)
            train_data = list(train_data)
            indice = indice_minimo_tramo
            
            while indice <= indice_maximo_tramo-3:
                train_data_lbl.append(data_lbl[indice])
                train_data.append(data[indice, 1:])
                #train_lbl_draw.append(data_lbl[indice])
                train_lbl_draw.append(dataOriginal_lbl[indice])
                yPredict_RFR_draw.append(data_lbl[indice])				
                indice += 1

            firts = True
            while indice <= indice_maximo_tramo:
                if firts:
                    firts = False
                    test_data_lbl = list()
                    test_data = list()
				
                    test_data_lbl.append(data_lbl[indice])
                    test_data.append(data[indice, 1:])

                else:
                    train_data_lbl=list(train_data_lbl)
                    train_data_lbl.append(yPredict_RFR[0])

                    train_data=list(train_data)
                    anio_aux = test_data[0, 0]    	#anio
                    aux_defle = test_data[0, 1] 	#deflexión
                    aux_tl = test_data[0, 2] 		#transito liviano
                    aux_tm = test_data[0, 3] 		#transito medio
                    aux_tp = test_data[0, 4] 		#transito pesado
                    aux_iri = test_data[0, 5]
                    train_data.append([anio_aux, aux_defle, aux_tl, aux_tm, aux_tp, aux_iri])
                    
                    test_data_lbl = list()
                    test_data = list()
                    test_data_lbl.append(data_lbl[indice])
                    test_data.append([data[indice, 1],data[indice, 2],data[indice, 3],data[indice, 4],data[indice, 5],yPredict_RFR[0]])
                
                test_data = np.array(test_data)
                test_data_lbl = np.array(test_data_lbl)
                #train_lbl_draw.append(data_lbl[indice])
                train_lbl_draw.append(dataOriginal_lbl[indice])				
                train_data_lbl = np.array(train_data_lbl)
                train_data = np.array(train_data)
				
                rfc = RFR(n_estimators=200, random_state=1, max_depth=4)
                rfc.fit(train_data, train_data_lbl)

                yPredict_RFR = rfc.predict(test_data)
                yPredict_RFR_draw.append(yPredict_RFR[0])
				
                indice += 1
            
			################## ENTRENAMOS ##################
            anioPred = 1
            n = test_data.shape
            m = train_data.shape
            size_test = n[0]
            size_train = m[0]
	   
            train_data_lbl=list(train_data_lbl)
            train_data_lbl.append(yPredict_RFR[0])

            train_data=list(train_data)
            anio_aux = test_data[0, 0]    	#anio
            aux_defle = test_data[0, 1] 	#deflexión
            aux_tl = test_data[0, 2] 		#transito liviano
            aux_tm = test_data[0, 3] 		#transito medio
            aux_tp = test_data[0, 4] 		#transito pesado
            aux_iri = test_data[0, 5]
            train_data.append([anio_aux, aux_defle, aux_tl, aux_tm, aux_tp, aux_iri])

            test_data_lbl = list()
            test_data = list()
            ind = indice_maximo_tramo
            tramo = data[ind, 0]
       
            anio = data[ind, 1]
            deflex = data[ind, 2]
            tP = data[ind, 3]*1.02
            tM = data[ind, 4]*1.02
            tL = data[ind, 5]*1.02
            iri = yPredict_RFR[0]
            test_data.append([anio+1, int(deflex), int(tP), int(tM), int(tL), iri])

            test_data = np.array(test_data)
            test_data_lbl = np.array(test_data_lbl)
            train_data_lbl = np.array(train_data_lbl)
            train_data = np.array(train_data)
            
            while anioPred <= 3:
                rfc = RFR(n_estimators=600, random_state=1, max_depth=7)
                rfc.fit(train_data, train_data_lbl)
			
                yPredict_RFR = rfc.predict(test_data)

                i = 0
                while i < size_test:
                    anio_aux = test_data[i, 0]    	#anio
                    aux_defle = test_data[i, 1] 	#deflexión
                    aux_tl = test_data[i, 2] 		#transito liviano
                    aux_tm = test_data[i, 3] 		#transito medio
                    aux_tp = test_data[i, 4] 		#transito pesado
                    aux_iri = yPredict_RFR[i]
            
                    test_data[i, 0] = test_data[i, 0] + 1   #anio
                    test_data[i, 1] = test_data[i, 1]       #deflexión
                    test_data[i, 2] = test_data[i, 2]*1.02  #transito pesado
                    test_data[i, 3] = test_data[i, 3]*1.02  #transito pesado
                    test_data[i, 4] = test_data[i, 4]*1.02  #transito pesado
                    test_data[i, 5] = yPredict_RFR[i]
		    
                    train_data_lbl=list(train_data_lbl)
                    train_data_lbl.append(yPredict_RFR[i])
		
                    train_data=list(train_data)
                    train_data.append([anio_aux, aux_defle, aux_tl, aux_tm, aux_tp, aux_iri])
	
                    i += 1			
			
                train_data = np.array(train_data)
                yPredict_RFR_draw.append(yPredict_RFR[0])				
        
                anioPred += 1
			
            print(tramoTrain)
			
            if args.plot:
                #Graficamos
                data_to_plot_dict = {"gt": train_lbl_draw, "rf": yPredict_RFR_draw}
                plotearContinuo_RFR.plot_personalizado(data_dict=data_to_plot_dict,
                           plot_title="Leave one out")
						   
    
if __name__ == "__main__":
    main(sys.argv[1:])
