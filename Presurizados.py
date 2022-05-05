from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('fast')

#usamos librería KERAS y TENSORFLOW
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, LSTM
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
#from keras.optimizers import Adam   #no se importa Adam


#CARGAMOS NUESTRO DATASET (sería, Fecha/Presión/Caudal)
df = pd.read_excel(r"C:\Users\t01hkruteler\Documents\PEO Abril-Sep\Presurizados ML\base ejemplo.xlsx", sheet_name = "1216")
#indexNames = df[ (df['Presión'] <10 )].index #eliminé lo datos muy pequeños que se salen de rango (VER)
#df.drop(indexNames , inplace=True)
df=df.round({"Qg":0, "Presión":0}) #redonde valores a cero decimales de presión y caudal
#del df['Qg']
print(df.head(20))
print(df.info())
print(df.describe())

#GRAFICO DE LOS DATOS
# fix, ax = plt.subplots(2,1,sharey=True)
# ax[0].plot(df['fecha'], df['Presión'])
# ax[1].plot(df['fecha'], df['Qg'], color='r')
# ax[0].set_xlabel("fecha")
# ax[1].set_xlabel("fecha")
# ax[0].set_ylabel("Presión")
# ax[1].set_ylabel("Caudal")
# plt.show()

#PROCESADO DE LOS DATOS
PASOS=7
#convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
values = df['Qg'].values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))

values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension

scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, PASOS, 1)
reframed.reset_index(inplace=True, drop=True)

contador=0
reframed['Estado']=df['Estado']
#reframed['month']=df['month']

for i in range(reframed.index[0],reframed.index[-1]):
    reframed['Estado'].loc[contador]=df['Estado'][i+8]
    #reframed['month'].loc[contador]=df['month'][i+8]
    contador=contador+1
#print(reframed.head())

reordenado=reframed[['Estado','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)','var1(t)']]
reordenado.dropna(inplace=True)
print(reordenado.tail())

#DIVIDIMOS EN SET DE ENTRENAMIENTO Y DE TESTEO/VALIDACIÓN
training_data = reordenado.drop('var1(t)',axis=1)#.values
target_data=reordenado['var1(t)']
#training_data.head()
valid_data = training_data[869-70:869]
valid_target=target_data[869-70:869]

training_data = training_data[0:869]
target_data=target_data[0:869]
print(training_data.shape,target_data.shape,valid_data.shape,valid_target.shape)
#print(training_data.head())

#CREAMOS EL MODELO DE RED NEURONAL
#Utilizaremos una Red Feedforward (Prealimentada) con Embeddings (INVESTIGAR)
#Tenenos como entradas 8 columnas (1 embeddings y 7 pasos)

def crear_modeloEmbeddings():
    emb_estado = 2 #tamanio profundidad de embeddings
    #emb_meses = 4

    in_estado = Input(shape=[1], name = 'Estado')
    emb_estado = Embedding(7+1, emb_estado)(in_estado)
    #in_meses = Input(shape=[1], name = 'meses')
    #emb_meses = Embedding(12+1, emb_meses)(in_meses)

    in_cli = Input(shape=[PASOS], name = 'cli')

    fe = concatenate([(emb_estado)])

    x = Flatten()(fe)
    x = Dense(PASOS,activation='tanh')(x) #TANGENTE HIPERBOLICA...VER SI ES EL ADECUADO
    outp = Dense(1,activation='tanh')(x)
    model = Model(inputs=[in_estado,in_cli], outputs=outp)

    model.compile(loss='mean_absolute_error', 
                  optimizer='adam',
                  metrics=['MSE'])

    model.summary()
    return model
    
#ENTRENAMOS NUESTRA MÁQUINA
EPOCHS=40

model = crear_modeloEmbeddings()

continuas=training_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
valid_continuas=valid_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]

history=model.fit([training_data['Estado'],continuas], target_data, epochs=EPOCHS
                 ,validation_data=([valid_data['Estado'],valid_continuas],valid_target))

results=model.predict([valid_data['Estado'],valid_continuas])
print( len(results) )
plt.scatter(range(len(valid_target)),valid_target,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
plt.show()

plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.show()


compara = pd.DataFrame(np.array([valid_target, [x[0] for x in results]])).transpose()
compara.columns = ['real', 'prediccion']

inverted = scaler.inverse_transform(compara.values)

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real', 'prediccion']
compara2['diferencia'] = compara2['real'] - compara2['prediccion']
print(compara2.head(40))

#PRONÓSTICO

