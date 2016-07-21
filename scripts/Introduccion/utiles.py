import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_hot_encoding(datos,caracteristicas_categoricas):    
    for cat in caracteristicas_categoricas:
        one_encoding=pd.get_dummies(datos[cat],prefix=cat)
        datos=pd.concat([datos,one_encoding],axis=1)
        del datos[cat]    
    return datos

def mostrar_imagenes(datos,target=None,prediccion=None):
    fig = plt.figure(figsize=(15, 3))
    n,_=datos.shape
    for i in range(n):
        titulo=''
        if not target is None:            
            titulo="T:{},".format(target[i])
        if not prediccion is None:
            titulo="{}P:{}".format(titulo,prediccion[i])

        ax = fig.add_subplot(1, n, 1 + i, xticks=[], yticks=[],title=titulo)
        ax.imshow(datos[i].reshape((8, 8)), cmap=plt.cm.binary)
        
