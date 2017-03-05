
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Activation, Dropout, Embedding, Masking, TimeDistributed, Merge, Lambda, Input, Permute, Reshape, merge
import numpy as np
from numpy import genfromtxt
from keras.preprocessing import sequence
from numpy import zeros, newaxis
import gc
from keras.models import load_model
from keras import backend as K
#from tqdm import tqdm
import theano as T

my_data1 = genfromtxt('C:/Users/lmglass/Documents/AHEdata.csv', delimiter=',',dtype=None)
my_data2 = genfromtxt('C:/Users/lmglass/Documents/AHEdataY.csv', delimiter=',',dtype=None)

my_data3 = genfromtxt('C:/Users/lmglass/Documents/AHEtest.csv', delimiter=',',dtype=None)
my_data4 = genfromtxt('C:/Users/lmglass/Documents/AHEtestY.csv', delimiter=',',dtype=None)

Xin = my_data1
Xin  = Xin [:,:,newaxis]

Xtest = my_data3 

Xtest = Xtest [:,:,newaxis]

y = my_data2 
y = y[:, newaxis]

ytest = my_data4 
ytest  = ytest [:, newaxis]

main_input = Input(shape=(Xin.shape[1],Xin.shape[2]), dtype='float32', name='main_input')
#mask1 = Masking(mask_value=0.)(main_input)
lstm_out1 = LSTM(10,return_sequences=True)(main_input )
#dropout1 = Dropout(0.2)(lstm_out1)
#lstm_out2 = LSTM(10,return_sequences=False, activation='sigmoid')(dropout1)
lstm_out2 = LSTM(10,return_sequences=False)(lstm_out1)
Dense_layer = Dense(1,init='normal')(lstm_out1 )

merged_model = Model(input=main_input, output=(Dense_layer))

#print(merged_model.predict([Xin, Xout],y).shape)

merged_model.compile(loss='binary_crossentropy', optimizer='adam')
merged_model.fit([Xin], y, nb_epoch=20, validation_split=0.1)

output = merged_model.predict([Xtest], verbose=0)
np.savetxt("C:/Users/lmglass/Documents/output20170303.csv", output, delimiter=",")
output = merged_model.predict([Xin], verbose=0)
np.savetxt("C:/Users/lmglass/Documents/output20170305.csv", output, delimiter=",")

#np.savetxt("C:/Users/lmglass/Documents/IMS Master Folder/Merger Workstreams/Site Ratings/Participation/y100.csv", Yout, delimiter=",")