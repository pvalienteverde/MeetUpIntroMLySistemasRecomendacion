# MeetUp Introduccion a MachineLearning: Sistemas de Recomendacion en python

Documentacion para el meetup de Reddadix Weekend: Introduccion a Machine Learning y Sistemas de Recomendacion en python.

La charla constará de dos partes:
* **Introduccion basica a Machine Learning utilizando sklearn:** Un breve repaso sobre los principales estimadores que nos provee sklearn.
* **Sistemas de recomendacion:**  Mostraremos como se pueden realizar los sistemas de recomendación aplicando tanto algoritmo heuristicos como machine learning

## Instalación de las dependencia utilizadas
Se ha usado las distribuciones facilitadas por el sistema, a continuación se describe  la preparación del entorno (distribución Fedora)

* **Python** Utilizamos para todo el paquete **Python3.X**. Si alguien utiliza Python2.X el cambio deberia ser trivial
* **Paquetes de python**: [pandas](http://pandas.pydata.org), [sklearn](http:/http://scikit-learn.org/stable/), [numpy](http://www.numpy.org), [bokeh](http://bokeh.pydata.org/en/latest) y [ipython](http://ipython.org/)

Es decir:

- Instalación de los paquetes de python

```bash
sudo dnf install python3-matplotlib python3-scipy python3-scikit-learn python3-pandas python3-pip python3-ipython-notebook
sudo pip3 install py4j seaborn nltk
```
- Ahora instalamos la última distribución de spark
```bash
cd $HOME
wget http://ftp.cixug.es/apache/spark/spark-1.6.2/spark-1.6.2-bin-hadoop2.6.tgz
tar xopf spark-1.6.2-bin-hadoop2.6.tgz
```
- Por último, incluimos en el fichero *~/.bash_profile*:
```bash
export SPARK_HOME=$HOME/spark-1.6.2-bin-hadoop2.6
export PYSPARK_SUBMIT_ARGS="--master local[*] pyspark-shell"
export PYSPARK_PYTHON=/usr/bin/ipython3
export PYSPARK_DRIVER_PYTHON=ipython3
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH
```

### Primeros pasos
Ya tenemos el entorno preparado, ahora vamos a lanzar nuestro pequeño script para ver que todo funciona bien:
```bash
git clone https://github.com/pvalienteverde/MeetUpIntroMLySistemasRecomendacion.git
cd MeetUpIntroMLySistemasRecomendacion/scripts
source ~/.bash_profile
python3 script_test.py
```


