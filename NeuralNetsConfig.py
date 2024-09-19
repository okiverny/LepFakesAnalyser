import  pandas as pd

### sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

### tensorflow libraries
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Dropout,AlphaDropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

### Flags
use_abs_weight = True

### Functions
features = []
def set_features(myfeatures):
    features = myfeatures

def prepare_data(data):

    df_qjet = data[data['tau_1_tmp']=='q-jet'].reset_index()
    df_gjet = data[data['tau_1_tmp']=='g-jet'].reset_index()

    df_qjet['class']=1
    df_gjet['class']=0 

    # Checking for balance
    print("Balancing samples...")
    #nevents_qjet = df_qjet.index[-1]
    #nevents_gjet = df_gjet.index[-1]
    nevents_qjet = df_qjet['weight_nominal'].sum()
    nevents_gjet = df_gjet['weight_nominal'].sum()
    print('Number of sig/bkg events before balance = ',nevents_qjet,'/', nevents_gjet,'( factor of ',nevents_gjet/nevents_qjet,')')

    #  Weight including balance
    df_qjet['weight_for_train'] = df_qjet.apply(lambda row: abs(row['weight_nominal'])*nevents_gjet/nevents_qjet, axis=1)
    df_gjet['weight_for_train'] = df_gjet.apply(lambda row: abs(row['weight_nominal']), axis=1)


    # Splitting into train/test
    df_qjet_train, df_qjet_test = train_test_split(df_qjet, train_size=0.7, shuffle=True)
    df_gjet_train, df_gjet_test = train_test_split(df_gjet, train_size=0.7, shuffle=True)
    df_train = pd.concat([df_qjet_train, df_gjet_train], axis=0)
    df_test = pd.concat([df_qjet_test, df_gjet_test], axis=0)

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    # Defining inputs
    list_of_branches_train = ['pt_lep1','eta_lep1','charge_lep1','type_lep1','isTight_lep1',
                          'pt_lep2','eta_lep2','charge_lep2','type_lep2','isTight_lep2',
                          'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack','fs_had_tau_1_tight',
                          'fs_had_tau_1_RNNScore','fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
                          'pt_jet1','pt_jet2']
    X_train, X_test = df_train[list_of_branches_train], df_test[list_of_branches_train]
    Y_train, Y_test = df_train['class'], df_test['class']
    weight_train, weight_test = df_train['weight_for_train'],df_test['weight_for_train']

    ## Scale Inputs
    X_scaler = StandardScaler() # or MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, weight_train, weight_test

### Model NN
def classifier_model(input_shape):
    # Input
    X_input = Input(input_shape)
    
    # Layer(s)
    X = AlphaDropout(rate=0.12)(X_input)
    X = Dense(64, activation="selu", 
              kernel_initializer='lecun_normal', # he_normal, he_uniform, lecun_uniform
              name = 'Dense1')(X)

    X = AlphaDropout(rate=0.10)(X)
    X = Dense(64, activation="selu", 
              kernel_initializer='lecun_normal',
              name = 'Dense2')(X)
    
    X = AlphaDropout(rate=0.10)(X)
    X = Dense(32, activation="selu", 
              kernel_initializer='lecun_normal',
              name = 'Dense3')(X)
    
    # Output
    X_output = Dense(1, activation='sigmoid', name='output_layer')(X)
    
    # Build model
    model = Model(inputs=X_input, outputs=X_output, name='classifier_model')

    return model

def classifier_model_simple(input_shape):
    # Input
    X_input = Input(input_shape)
    
    # Layer(s)
    X = Dropout(rate=0.12)(X_input)
    X = Dense(64, activation="relu", 
              #kernel_initializer='lecun_normal', # he_normal, he_uniform, lecun_uniform
              name = 'Dense1')(X)

    X = Dropout(rate=0.10)(X)
    X = Dense(64, activation="relu", 
              #kernel_initializer='lecun_normal',
              name = 'Dense2')(X)
    
    X = Dropout(rate=0.10)(X)
    X = Dense(32, activation="relu", 
              #kernel_initializer='lecun_normal',
              name = 'Dense3')(X)
    
    # Output
    X_output = Dense(1, activation='sigmoid', name='output_layer')(X)
    
    # Build model
    model = Model(inputs=X_input, outputs=X_output, name='classifier_model')

    return model


