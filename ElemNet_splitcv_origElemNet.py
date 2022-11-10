import numpy as np
np.random.seed(1234567)
import tensorflow as tf
tf.random.set_seed(1234567)
import random
random.seed(1234567)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.layers import Dropout, AlphaDropout, GaussianDropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import average
from collections import Counter
from tensorflow.keras.layers import Input
import re, os, csv, math, operator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation

import argparse

parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("-es", "--early_stopping", help="Prints the supplied argument.", default=10, type=int)
parser.add_argument("-sm", "--saved_model", help="Prints the supplied argument.", default=None, type=str)
parser.add_argument("-prop", "--property", help="Prints the supplied argument.", default=None, type=str)


args = parser.parse_args()

es = args.early_stopping
sm = args.saved_model
prop = args.property

print("es:", es)
print("sm:", sm)

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu' ]

elements_all = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
                'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']

phys_atts = ['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm', 'MagpieData minimum Number', 'MagpieData maximum Number', 'MagpieData range Number', 'MagpieData mean Number', 'MagpieData avg_dev Number', 'MagpieData mode Number', 'MagpieData minimum MendeleevNumber', 'MagpieData maximum MendeleevNumber', 'MagpieData range MendeleevNumber', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode MendeleevNumber', 'MagpieData minimum AtomicWeight', 'MagpieData maximum AtomicWeight', 'MagpieData range AtomicWeight', 'MagpieData mean AtomicWeight', 'MagpieData avg_dev AtomicWeight', 'MagpieData mode AtomicWeight', 'MagpieData minimum MeltingT', 'MagpieData maximum MeltingT', 'MagpieData range MeltingT', 'MagpieData mean MeltingT', 'MagpieData avg_dev MeltingT', 'MagpieData mode MeltingT', 'MagpieData minimum Column', 'MagpieData maximum Column', 'MagpieData range Column', 'MagpieData mean Column', 'MagpieData avg_dev Column', 'MagpieData mode Column', 'MagpieData minimum Row', 'MagpieData maximum Row', 'MagpieData range Row', 'MagpieData mean Row', 'MagpieData avg_dev Row', 'MagpieData mode Row', 'MagpieData minimum CovalentRadius', 'MagpieData maximum CovalentRadius', 'MagpieData range CovalentRadius', 'MagpieData mean CovalentRadius', 'MagpieData avg_dev CovalentRadius', 'MagpieData mode CovalentRadius', 'MagpieData minimum Electronegativity', 'MagpieData maximum Electronegativity', 'MagpieData range Electronegativity', 'MagpieData mean Electronegativity', 'MagpieData avg_dev Electronegativity', 'MagpieData mode Electronegativity', 'MagpieData minimum NsValence', 'MagpieData maximum NsValence', 'MagpieData range NsValence', 'MagpieData mean NsValence', 'MagpieData avg_dev NsValence', 'MagpieData mode NsValence', 'MagpieData minimum NpValence', 'MagpieData maximum NpValence', 'MagpieData range NpValence', 'MagpieData mean NpValence', 'MagpieData avg_dev NpValence', 'MagpieData mode NpValence', 'MagpieData minimum NdValence', 'MagpieData maximum NdValence', 'MagpieData range NdValence', 'MagpieData mean NdValence', 'MagpieData avg_dev NdValence', 'MagpieData mode NdValence', 'MagpieData minimum NfValence', 'MagpieData maximum NfValence', 'MagpieData range NfValence', 'MagpieData mean NfValence', 'MagpieData avg_dev NfValence', 'MagpieData mode NfValence', 'MagpieData minimum NValence', 'MagpieData maximum NValence', 'MagpieData range NValence', 'MagpieData mean NValence', 'MagpieData avg_dev NValence', 'MagpieData mode NValence', 'MagpieData minimum NsUnfilled', 'MagpieData maximum NsUnfilled', 'MagpieData range NsUnfilled', 'MagpieData mean NsUnfilled', 'MagpieData avg_dev NsUnfilled', 'MagpieData mode NsUnfilled', 'MagpieData minimum NpUnfilled', 'MagpieData maximum NpUnfilled', 'MagpieData range NpUnfilled', 'MagpieData mean NpUnfilled', 'MagpieData avg_dev NpUnfilled', 'MagpieData mode NpUnfilled', 'MagpieData minimum NdUnfilled', 'MagpieData maximum NdUnfilled', 'MagpieData range NdUnfilled', 'MagpieData mean NdUnfilled', 'MagpieData avg_dev NdUnfilled', 'MagpieData mode NdUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData maximum NfUnfilled', 'MagpieData range NfUnfilled', 'MagpieData mean NfUnfilled', 'MagpieData avg_dev NfUnfilled', 'MagpieData mode NfUnfilled', 'MagpieData minimum NUnfilled', 'MagpieData maximum NUnfilled', 'MagpieData range NUnfilled', 'MagpieData mean NUnfilled', 'MagpieData avg_dev NUnfilled', 'MagpieData mode NUnfilled', 'MagpieData minimum GSvolume_pa', 'MagpieData maximum GSvolume_pa', 'MagpieData range GSvolume_pa', 'MagpieData mean GSvolume_pa', 'MagpieData avg_dev GSvolume_pa', 'MagpieData mode GSvolume_pa', 'MagpieData minimum GSbandgap', 'MagpieData maximum GSbandgap', 'MagpieData range GSbandgap', 'MagpieData mean GSbandgap', 'MagpieData avg_dev GSbandgap', 'MagpieData mode GSbandgap', 'MagpieData minimum GSmagmom', 'MagpieData maximum GSmagmom', 'MagpieData range GSmagmom', 'MagpieData mean GSmagmom', 'MagpieData avg_dev GSmagmom', 'MagpieData mode GSmagmom', 'MagpieData minimum SpaceGroupNumber', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range SpaceGroupNumber', 'MagpieData mean SpaceGroupNumber', 'MagpieData avg_dev SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber', 'avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons', 'avg f valence electrons', 'compound possible', 'max ionic char', 'avg ionic char']
# Regex to Choose from Element Name, Number and Either of the Brackets
token = re.compile('[A-Z][a-z]?|\d+|[()]')

# Create a dictionary with the Name of the Element as Key and No. of elements as Value
def count_elements(formula):
    tokens = token.findall(str(formula))
    stack = [[]]
    for t in tokens:
        if t.isalpha():
            last = [t]
            stack[-1].append(t)
        elif t.isdigit():
             stack[-1].extend(last*(int(t)-1))
        elif t == '(':
            stack.append([])
        elif t == ')':
            last = stack.pop()
            stack[-1].extend(last)   
    return dict(Counter(stack[-1]))

#Normalize the Value of the Dictionary
def normalize_elements(dictionary):
    dic_val = sum(dictionary.values()) 
    if dic_val == 0:
        factor = 0
    else:    
        factor=1.0/ dic_val  
        
    for k in dictionary:
        dictionary[k] = dictionary[k]*factor
    return dictionary

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

print(Diff(elements_all, elements)) 

def elemental_fraction(dataframe):
    print('The loaded dataset has %d entries'%len(dataframe['pretty_comp']))

    #data = mp_data[mp_data['composition'].str.contains('(H|Li|Be|B|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Ac|Th|Pa|U|Np|Pu)[0-9]+', case=True)]
    #data = dataframe[~dataframe['pretty_comp'].str.contains('Bk|Md|Ds|Sg|Ar|No|At|Db|He|Po|Fr|Cm|Cn|Rn|Mt|Fm|Cf|Hs|Ra|Es|Bh|Rf|Lr|Rg|Ne|Am')]
    compounds = dataframe['pretty_comp']

    print('The reduced dataset has %d entries'%len(compounds))
    
    compounds = [count_elements(x) for x in compounds]
    compounds = [normalize_elements(x) for x in compounds]

    in_elements = np.zeros(shape=(len(compounds), len(elements)))
    comp_no = 0

    for compound in compounds:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  
    
    data = in_elements
    
    return data

train = pd.read_csv(r'/data/vgf3011/doubletldata/doubletl_pero/doubletl_pero_train_123456789.csv') 
val = pd.read_csv(r'/data/vgf3011/doubletldata/doubletl_pero/doubletl_pero_set_10.csv') 
test = pd.read_csv(r'/data/vgf3011/doubletldata/doubletl_pero/doubletl_pero_test_set.csv') 
#oqmd = oqmd[~oqmd['formulae'].str.contains('Bk|Md|Ds|Sg|Ar|No|At|Db|He|Po|Fr|Cm|Cn|Rn|Mt|Fm|Cf|Hs|Ra|Es|Bh|Rf|Lr|Rg|Ne|Am', na=False)]

print("done loading")  

prop = '{}'.format(prop)

train = train[train[prop].notnull()]
val = val[val[prop].notnull()]
test = test[test[prop].notnull()]

print(train.shape)
print(val.shape)
print(test.shape)

print(prop)



x_train = train.pop('pretty_comp').to_frame()
y_train = train.pop(prop).to_frame()
x_val = val.pop('pretty_comp').to_frame()
y_val = val.pop(prop).to_frame()
x_test = test.pop('pretty_comp').to_frame()
y_test = test.pop(prop).to_frame()

new_x_train = elemental_fraction(x_train)
new_x_val = elemental_fraction(x_val)
new_x_test = elemental_fraction(x_test)

new_y_train = np.array(y_train)
new_y_val = np.array(y_val)
new_y_test = np.array(y_test)

new_y_train.shape = (len(new_y_train),)
new_y_val.shape = (len(new_y_val),)
new_y_test.shape = (len(new_y_test),)


in_layer = Input(shape=(86,))

layer_1 = Dense(1024)(in_layer)
layer_1 = Activation('relu')(layer_1)

layer_2 = Dense(1024)(layer_1)
layer_2 = Activation('relu')(layer_2)

layer_3 = Dense(1024)(layer_2)
layer_3 = Activation('relu')(layer_3)

layer_4 = Dense(1024)(layer_3)
layer_4 = Activation('relu')(layer_4)
layer_4 = Dropout(0.2)(layer_4, training=True)

layer_5 = Dense(512)(layer_4)
layer_5 = Activation('relu')(layer_5)

layer_6 = Dense(512)(layer_5)
layer_6 = Activation('relu')(layer_6)

layer_7 = Dense(512)(layer_6)
layer_7 = Activation('relu')(layer_7)
layer_7 = Dropout(0.1)(layer_7, training=True)

layer_8 = Dense(256)(layer_7)
layer_8 = Activation('relu')(layer_8)

layer_9 = Dense(256)(layer_8)
layer_9 = Activation('relu')(layer_9)

layer_10 = Dense(256)(layer_9)
layer_10 = Activation('relu')(layer_10)
layer_10 = Dropout(0.3)(layer_10, training=True)

layer_11 = Dense(128)(layer_10)
layer_11 = Activation('relu')(layer_11)

layer_12 = Dense(128)(layer_11)
layer_12 = Activation('relu')(layer_12)

layer_13 = Dense(128)(layer_12)
layer_13 = Activation('relu')(layer_13)
layer_13 = Dropout(0.2)(layer_13, training=True)

layer_14 = Dense(64)(layer_13)
layer_14 = Activation('relu')(layer_14)

layer_15 = Dense(64)(layer_14)
layer_15 = Activation('relu')(layer_15)

layer_16 = Dense(32)(layer_15)
layer_16 = Activation('relu')(layer_16)

out_layer = Dense(1)(layer_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es, restore_best_weights=True)

# Fit the model
model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_val, new_y_val), epochs=3000, batch_size=32, callbacks=[es])

results = model.evaluate(new_x_test, new_y_test, batch_size=32)
print(results)

model_json = model.to_json()
with open("{}.json".format(sm), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("{}.h5".format(sm))
print("Saved model to disk")