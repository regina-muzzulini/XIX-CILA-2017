import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_personalizado(data_dict, plot_title):
    """

    :param data_dict: diccionario con datos
    :param plot_title: título de la gráfica


    :return:
    """

    MARKER_SIZE = 9
    LINE_WIDTH = 1.7
    MARKER='o'
    gt_data = data_dict["gt"]
    rf_data = data_dict["rf"]
			 
    plt.plot(rf_data,
             c='fuchsia',
             marker=MARKER,
             fillstyle='full',
             label="RFR",
             linewidth=LINE_WIDTH,
             markeredgecolor='yellow',
             markersize=MARKER_SIZE)
    
    plt.plot(gt_data, c='maroon', marker=MARKER, fillstyle='full', label='valores reales del tramo',
             linewidth = LINE_WIDTH,
             markersize = MARKER_SIZE,
             markeredgecolor='yellow'
             )

    plt.xlabel("Años", fontsize=16)  # Insertamos el titulo del eje X
    plt.ylabel("Rugosidad [m/km]", fontsize=16, rotation="90")  # Insertamos el titulo del eje Y

    plt.yticks(np.arange(1.5, 3.5, 0.5))
	
    plt.grid(alpha=0.6)

    plt.title(plot_title, fontsize=18)
    plt.legend(loc=4, fontsize=12)
    plt.show(block=True)