from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from math import sqrt
import pandas as pd
import numpy as np
from operator import add

conf = (SparkConf()
         .setMaster("local[4]")
         .setAppName("Myapp")
         .set("spark.executor.memory", "2g"))
sc = SparkContext(conf = conf)

sc.setLogLevel("ERROR")

matriz_datos=np.array([(0, 0, 4.0), 
                                        (0, 1, 2.0), 
                                        (0, 5, 5.0), 
                                        (0, 4, 4.0), 
                                        (1, 1, 3.0), 
                                        (1, 2, 4.0), 
                                        (1, 5, 5.0), 
                                        (1, 4, 4.0), 
                                        (2, 1, 1.0), 
                                        (2, 2, 5.0)])

datos_df=pd.DataFrame(data=matriz_datos,columns=["user", "item", "rating"])


sqlContext = SQLContext(sc)
dfRatings = sqlContext.createDataFrame(datos_df)

from pyspark.ml.evaluation import Evaluator

class MiEvaluador(Evaluator):

    def __init__(self,predictionCol='prediction', targetCol='rating'):        
        super(MiEvaluador, self).__init__()
        self.predictionCol=predictionCol
        self.targetCol=targetCol
        
    def _evaluate(self, dataset):       
        error=self.rmse(dataset,self.predictionCol,self.targetCol)
        print ("Error: {}".format(error))
        return error
    
    def isLargerBetter(self):
        return False
    
    @staticmethod
    def rmse(dataset,predictionCol,targetCol):
        return sqrt(dataset.dropna().map(lambda x: (x[targetCol] - x[predictionCol]) ** 2).reduce(add) / float(dataset.count()))


    
lr1 = ALS()
grid1 = ParamGridBuilder().addGrid(lr1.regParam, [1.0,0.5,2.0]).build()
evaluator1 = MiEvaluador(predictionCol=lr1.getPredictionCol(),targetCol=lr1.getRatingCol())
cv1 = CrossValidator(estimator=lr1, estimatorParamMaps=grid1, evaluator=evaluator1, numFolds=2)
cvModel1 = cv1.fit(dfRatings)
a=cvModel1.transform(dfRatings)
error_cross_validation=MiEvaluador.rmse(a,lr1.getPredictionCol(),lr1.getRatingCol())
print ('ERROR de validacion: {}'.format(error_cross_validation))

error_models=[]
for reg_param in (1.0,0.5,2.0):
    lr = ALS(regParam=reg_param)
    model = lr.fit(dfRatings)
    error=MiEvaluador.rmse(model.transform(dfRatings),lr.getPredictionCol(),lr.getRatingCol())
    error_models.append(error)
    print ('reg_param: {}, rmse: {}'.format(reg_param,error))
    
import numpy as np
if np.isclose(error_models[np.argmin(error_models)],error_cross_validation):
	print("***\nFunciona correctamente pyspark\n****")	
else:
	raise RuntimeError("Deberia coincidir con el modulo donde reg_param = 0.5")
