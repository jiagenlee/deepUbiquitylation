import os
import theano
import time
import numpy as np
import pandas as pd
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import merge
from keras.layers import pooling
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU,ThresholdedReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback,LearningRateScheduler,History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras import backend as K
import keras.metrics
import copy,math

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

def mcc(y_true,y_pred):
    
    y_pred_label = K.round(K.clip(y_pred, 0, 1))
    y_true_label = K.round(K.clip(y_true, 0, 1))
    
    tp = K.sum(y_true_label * y_pred_label)
    fn = K.sum(y_true_label * (1.0 - y_pred_label))
    fp = K.sum((1.0 - y_true_label) * y_pred_label)
    tn = K.sum((1.0- y_true_label) * (1.0 - y_pred_label))

    numerator = tp * tn - fp * fn
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / denominator

    
def mcc_loss(y_true,y_pred):
    
    return K.mean (1 / K.exp(mcc(y_true, y_pred)))

def f1_score(y_true,y_pred):

    y_pred_label = K.round(K.clip(y_pred, 0, 1))
    y_true_label = K.round(K.clip(y_true, 0, 1))
    
    tp = K.sum(y_true_label * y_pred_label)
    fn = K.sum(y_true_label * (1.0 - y_pred_label))
    fp = K.sum((1.0 - y_true_label) * y_pred_label)
    
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn
    
    return numerator / denominator 

def f1_loss(y_true,y_pred):
    
    return K.mean ((-1) * K.log(f1_score(y_true, y_pred)))

########## MultiCNN ##########
def MultiCNN(trainX,trainY,trainPhysicalX,train_pssmX,
             pre_train_seq_path = None,
             pre_train_physical_path = None,
             pre_train_pssm_path = None,
             nb_epoch=1000,earlystop=None,transferlayer=1,weights=None,forkinas=False,compiletimes=0,compilemodels=None,
             batch_size=2048,    
             class_weights = {0 : 0.5 , 1 : 1},
             predict=False):

    ########## Set Oneofkey Network Size and Data ##########
    input_row  = trainX.shape[2] 
    input_col  = trainX.shape[3] 
    trainX_t=trainX; 
    
    ########## Set Physical Network Size and Data ##########
    physical_row = trainPhysicalX.shape[2] 
    physical_col = trainPhysicalX.shape[3]
    train_physical_X_t = trainPhysicalX 
    
    ########## Set Pssm Network Size and Data ##########
    pssm_row = train_pssmX.shape[2] 
    pssm_col = train_pssmX.shape[3]
    train_pssm_X_t = train_pssmX

    ########## Set Early_stopping ##########
    if(earlystop is not None): 
        early_stopping = EarlyStopping(monitor='val_loss',mode='min', patience = 20)
    nb_epoch = 500;#set to a very big value since earlystop used
    
    ########## TrainX_t For Shape ##########
    trainX_t.shape=(trainX_t.shape[0],input_row,input_col) 
    input = Input(shape=(input_row,input_col))

    ########## Train_physical_X_t For Shape ##########
    train_physical_X_t.shape = (train_physical_X_t.shape[0],physical_row,physical_col)
    physicalInput = Input(shape=(physical_row,physical_col))
    
    ########## Train_pssm_X_t For Shape ##########
    train_pssm_X_t.shape = (train_pssm_X_t.shape[0],pssm_row,pssm_col)
    pssmInput = Input(shape=(pssm_row,pssm_col)) 
    
    if compiletimes==0:         
        
        ########## Total Set Classes ##########
        nb_classes=2
        
        ########## Total Set Batch_size ##########
        batch_size=2048

        ########## Total Set Optimizer ##########
        #optimizer = SGD(lr=0.0001, momentum=0.9, nesterov= True)
        optimization='Nadam';

        ########## Begin Oneofkey Network ##########
        x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '0')(input)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)
    
        x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name = '1')(x)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)
    
        x = conv.Convolution1D(151,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name = '2')(x)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)
    
        x = conv.Convolution1D(101,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name = '3')(x)
        x = Activation('softsign')(x)                   
        x_reshape = core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
        x = Dropout(0.4)(x)

        output_x = core.Flatten()(x)
        output = BatchNormalization()(output_x)
        output = Dropout(0)(output)
        
        output = Dense(256,init='glorot_normal',activation='relu',name = '4')(output)
        output = Dropout(0.298224)(output)
        output = Dense(128,init='glorot_normal',activation="relu",name = '5')(output)
        output = Dropout(0)(output)
        output = Dense(128,activation="relu",init='glorot_normal',name = '6')(output)
        ########## End Oneofkey Network ##########
     
        ########## Begin Physical Network ##########
        physical_code_x = core.Flatten()(physicalInput)
        physical_code_x = BatchNormalization()(physical_code_x)
        
        physical_code_x = Dense(1024,init='glorot_normal',activation='softplus',name = '7')(physical_code_x)
        physical_code_x = BatchNormalization()(physical_code_x)
        physical_code_x = Dropout(0.2)(physical_code_x)
        
        physical_code_x = Dense(512,init='glorot_normal',activation='softplus',name = '8')(physical_code_x) 
        physical_code_x = BatchNormalization()(physical_code_x)
        physical_code_x = Dropout(0.4)(physical_code_x)
        
        physical_code_x = Dense(256,init='glorot_normal',activation='softplus',name = '9')(physical_code_x)
        physical_code_x = BatchNormalization()(physical_code_x)
        physical_code_x = Dropout(0.5)(physical_code_x)
        
        output_physical_x = Dense(128,init='glorot_normal',activation='relu',name = '10')(physical_code_x)
        ########## End Physical Network ##########
    
        ########## Begin Pssm Network ##########      
        pssm_x = conv.Convolution1D(200,1,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '11')(pssmInput)
        pssm_x = Activation('relu')(pssm_x)
        pssm_x = Dropout(0.5)(pssm_x)
        
        pssm_x = conv.Convolution1D(150,8,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '12')(pssm_x)
        pssm_x = Activation('relu')(pssm_x)   
        pssm_x = Dropout(0.5)(pssm_x)
        
        pssm_x = conv.Convolution1D(200,9,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '13')(pssm_x)
        pssm_x = Activation('relu')(pssm_x)
        pssm_x = Dropout(0.5)(pssm_x)        
        
        pssm_x_reshape1 = core.Reshape((pssm_col,pssm_row))(pssmInput)
        pssm_x_reshape2 = conv.Convolution1D(200,1,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '14')(pssm_x_reshape1)
        pssm_x_reshape2 = Activation('relu')(pssm_x_reshape2)
        pssm_x_reshape2 = Dropout(0.5)(pssm_x_reshape2) 
              
        pssm_x_reshape2 = conv.Convolution1D(150,3,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '15')(pssm_x_reshape2)
        pssm_x_reshape2 = Activation('relu')(pssm_x_reshape2) 
        pssm_x_reshape2 = Dropout(0.5)(pssm_x_reshape2)        
         
        pssm_x_reshape2 = conv.Convolution1D(200,7,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name = '16')(pssm_x_reshape2)
        pssm_x_reshape2 = Activation('relu')(pssm_x_reshape2)
        pssm_x_reshape2 = Dropout(0.5)(pssm_x_reshape2)        
        
        pssm_x = core.Flatten()(pssm_x)
        pssm_x_reshape2 = core.Flatten()(pssm_x_reshape2)
       
        pssm_output = merge([pssm_x,pssm_x_reshape2],mode='concat')
        pssm_output = Dropout(0)(pssm_output)
        
        pssm_output = BatchNormalization()(pssm_output)
        
        pssm_output=Dense(128,init='glorot_normal',activation='relu',name = '17')(pssm_output)
        pssm_output=Dropout(0.298224)(pssm_output)
        pssm_output=Dense(128,init='glorot_normal',activation='relu',name = '18')(pssm_output)
        pssm_output=Dropout(0)(pssm_output)        
        ########## End Pssm Network ##########
    
        ########## Set Output For Merge ########## 
        output = merge([output,output_physical_x,pssm_output],mode='concat')
        
        ########## Total Network After Merge ##########
        '''
        output = Dense(512,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '16')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.5)(output)
        output = Dense(256,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '17')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.3)(output)  
        output = Dense(49,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '18')(output)
        output = BatchNormalization()(output)
        output = Dropout(0)(output)
        '''          
        out = Dense(nb_classes,init='glorot_normal',activation='softmax',W_regularizer= l2(0.001),name = '19')(output)
        ########## Total Network End ##########
    
        ########## Set Cnn ##########
        cnn=Model([input,physicalInput,pssmInput],out)
        cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
        
        ########## Load Models ##########
        
        if (pre_train_seq_path is not None):
            seq_model = models.load_model(pre_train_seq_path) 
            for l in range(0, 6): #the last layers is not included
                cnn.get_layer(name = str(l)).set_weights(seq_model.get_layer(name = str(l)).get_weights()) 
                cnn.get_layer(name = str(l)).trainable = False
            #cnn.get_layer()
        
        if (pre_train_physical_path is not None):
            physical_model = models.load_model(pre_train_physical_path) 
            for l in range(7, 10):
                #len(seq_model.layers), (len(seq_model.layers)+len(physical_model.layers)-1)): #the last layer is not included
                cnn.get_layer(name = str(l)).set_weights(physical_model.get_layer(name = str(l)).get_weights())
                cnn.get_layer(name = str(l)).trainable = False
        
        if (pre_train_pssm_path is not None):
            pssm_model = models.load_model(pre_train_pssm_path) 
            for l in range(11, 18):
                #len(seq_model.layers), (len(seq_model.layers)+len(pssm_model.layers)-1)): #the last layer is not included
                cnn.get_layer(name = str(l)).set_weights(pssm_model.get_layer(name = str(l)).get_weights())
                cnn.get_layer(name = str(l)).trainable = False
        
    else:
        cnn=compilemodels
    
    ########## Set Class_weight ##########
    #oneofkclass_weights={0 : 0.78 , 1 : 1}
    #physicalclass_weights={0 : 0.3 , 1 : 1}
    #pssmclass_weights={0 : 0.95 , 1 : 1}
    #totalclass_weights={0 : 0.1 , 1 : 1}    
    
    if(predict is False):      
        if(trainY is not None):
            if(earlystop is None):
                #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(valX_t, valY))
                #fitHistory = cnn.fit(train_physical_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(val_physical_x_t, valY))                
                #fitHistory = cnn.fit(train_pssm_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(val_pssm_x_t, valY))
                fitHistory = cnn.fit([trainX_t,train_physical_X_t,train_pssm_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([valX_t,val_physical_x_t,val_pssm_x_t], valY))
            else:
                #checkpointer = ModelCheckpoint(filepath='oneofk.h5',verbose=1,save_best_only=True)
                #weight_checkpointer = ModelCheckpoint(filepath='oneofkweight.h5',verbose=1,save_best_only=True,monitor='val_acc',mode='max',save_weights_only=True)                
                #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, shuffle= True, validation_split=0.2, callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight = oneofkclass_weights)
                
                #checkpointer = ModelCheckpoint(filepath='physical.h5',verbose=1,save_best_only=True,monitor='val_acc',mode='max')
                #weight_checkpointer = ModelCheckpoint(filepath='physicalweight.h5',verbose=1,save_best_only=True,monitor='val_acc',mode='max',save_weights_only=True)
                #fitHistory = cnn.fit(train_physical_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, shuffle= True, validation_split=0.2, callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight = physicalclass_weights)                
                
                #checkpointer = ModelCheckpoint(filepath='pssm.h5',verbose=1,save_best_only=True,monitor='val_acc',mode='max')
                #weight_checkpointer = ModelCheckpoint(filepath='pssmweight.h5',verbose=1,save_best_only=True,monitor='val_acc',mode='max',save_weights_only=True)
                #fitHistory = cnn.fit(train_pssm_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, shuffle= True, validation_split=0.2, callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight = pssmclass_weights)
                
                checkpointer = ModelCheckpoint(filepath=str(compiletimes) + '-'+ str(class_weights[0]) +'-'+'merge.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
                weight_checkpointer = ModelCheckpoint(filepath=str(compiletimes) + '-'+ str(class_weights[0]) +'-'+'mergeweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
                fitHistory = cnn.fit([trainX_t,train_physical_X_t,train_pssm_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch, shuffle= True, validation_split=0.4, callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight = class_weights)
                
                #with open('siqingaowa.txt','a') as f:
                    #f.write(str(fitHistory.history)) 
                #f.close();                
        else:
            #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
            #fitHistory = cnn.fit(train_physical_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
            #fitHistory = cnn.fit(train_pssm_X_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
            fitHistory = cnn.fit([trainX_t,train_physical_X_t,train_pssm_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch)
    return cnn