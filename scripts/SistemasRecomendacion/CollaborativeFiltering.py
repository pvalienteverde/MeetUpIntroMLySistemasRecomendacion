import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pyspark.ml.evaluation import RegressionEvaluator,Evaluator
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

class EvaluadorRMSE(Evaluator):
    """
    Evalua RMSE de forma robusta.
    Es Igual que RegressionEvaluator con metric=rmse pero descartando valores no predecidos
    
    """
    def __init__(self,predictionCol, targetCol):        
        super(EvaluadorRMSE, self).__init__()
        self.predictionCol=predictionCol
        self.targetCol=targetCol
        
    def _evaluate(self, dataset):       
        error=rmse(dataset,self.predictionCol,self.targetCol)
        print ("Error: {}".format(error))
        return error
    
    def isLargerBetter(self):
        return False
    
    
class ModelBasedALS(object):
    """
    Envoltorio para la clase ALS de ml de Spark. 
    Da soporte a los metodos de ALS de mllib
    
    """
    def __init__(self,modelALS):
        super(ModelBasedALS, self).__init__()
        """
        Parametros
        ----------
        modelALS : objeto entrenado de pyspark.ml.recommendation.ALS
        """
        self.userIndex,self.userFactors = self.toArray(modelALS.userFactors)
        self.itemIndex,self.itemFactors = self.toArray(modelALS.itemFactors)
        self.prediccion=pd.DataFrame(data=self.userFactors.dot(self.itemFactors.T),columns=self.itemIndex,index=self.userIndex)
        
        self.relacion_index_user=dict(zip(self.userIndex,range(len(self.userIndex))))
        self.relacion_index_item=dict(zip(self.itemIndex,range(len(self.itemIndex))))        


    def predictAll(self,user_item:pd.DataFrame,tag_prediccion='prediccion'):
        """
        Devuelve todas las predicciones dado el par (user,item)
        """
        estimaciones=[]
        for tupla in user_item.values:
            try:
                estimacion=self.prediccion.iloc[self.relacion_index_user[tupla[0]],self.relacion_index_item[tupla[1]]]
                estimaciones.append(estimacion)
            except:
                estimaciones.append(np.nan)

        user_item[tag_prediccion]=estimaciones
        return user_item
    
    def recommendProducts(self,user:int,n:int=3):
        """
        Devuelve el top de productos recomendados para el usuario
        """
        usuario=self.prediccion.loc[user]
        usuario.sort(ascending=False)
        return usuario.iloc[:n]
    
    def recommendUsers(self,product:int,n:int=3):
        """
        Devuelve el top de los usuarios de un producto
        """
        productos=self.prediccion.loc[:,product]
        productos.sort(ascending=False)        
        return productos.iloc[:n]
        
    @staticmethod
    def toArray(datos):
        indices=[]
        lista=[]
        aaa=datos.rdd.map(lambda l:(l.id,l.features)).collect()
        for tupla in aaa:
            indices.append(tupla[0])
            lista.append(tupla[1])

        return indices,np.array(lista)    
    
   
   
    
    
def rmse(dataset,predictionCol,targetCol):
    valores=np.array(dataset.dropna().map(lambda r:[r[predictionCol],r[targetCol]]).collect())
    error = sqrt(mean_squared_error(valores[:,0],valores[:,1]))
    return error
