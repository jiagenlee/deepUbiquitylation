import sys
import os
import pandas as pd
import numpy as np
import argparse
import keras.metrics
from getData import get_data
from DProcess import convertRawToXY
from multiCNN import MultiCNN
import pickle,random
from keras import backend as K
import theano
import keras.utils.np_utils as kutils
import sklearn.metrics
import pickle,random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import csv

def main(srate=1,nb_epoch1=1,nb_epoch2=30,earlystop=20,
        maxneg=None,codingMode=0,transferlayer=1,inputweights=None,
        outputweights=None,forkinas=False):
    
    ########## Load Training Data ##########
    oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,physical_pos,physical_neg = get_data(r'data/Ubisite_train3.txt',r'data/pssmpickle2/',label = True) 

    ########## Load Testing Data ########## 
    test_oneofkey_pos,test_oneofkey_neg,test_pssm_pos,test_pssm_neg,test_physical_pos,test_physical_neg = get_data(r'data/Ubisite_test3.txt',r'data/pssmpickle2/',label = False)

    ########## Oneofkey Testing ##########
    test_oneofkey_pos = pd.DataFrame(test_oneofkey_pos)
    test_oneofkey_neg = pd.DataFrame(test_oneofkey_neg)
    test_oneofkey_all = pd.concat([test_oneofkey_pos,test_oneofkey_neg])
    test_oneofkeyX,test_oneofkeyY = convertRawToXY(test_oneofkey_all.as_matrix(),codingMode=0)
    
    ########## Physical Testing ##########
    test_physical_pos = pd.DataFrame(test_physical_pos)
    test_physical_neg = pd.DataFrame(test_physical_neg)
    test_physical_all = pd.concat([test_physical_pos,test_physical_neg])
    test_physicalX,test_physicalY = convertRawToXY(test_physical_all.as_matrix(),codingMode=6) 
    
    ########## Pssm Testing ##########
    test_pssm_all = test_pssm_pos+test_pssm_neg
    test_pssmX = convertRawToXY(test_pssm_all,codingMode=7)
    test_pssmY = test_oneofkeyY
    
    ########## OneofkeyX_t For Shape ##########
    test_oneofkeyX_t = test_oneofkeyX
    test_oneofkeyX_t.shape = (test_oneofkeyX.shape[0],test_oneofkeyX.shape[2],test_oneofkeyX.shape[3])
    
    ########## PhysicalX_t For Shape ##########
    test_physicalX_t = test_physicalX
    test_physicalX_t.shape = (test_physicalX.shape[0],test_physicalX.shape[2],test_physicalX.shape[3])        
    
    ########### PssmX_t For Shape ##########
    testPssmX_t = test_pssmX   
    testPssmX_t.shape=(test_pssmX.shape[0],test_pssmX.shape[2],test_pssmX.shape[3])     
    
    ########## Del Testall ##########
    del test_oneofkey_all,test_physical_all,test_pssm_all
    
    ########## Set Training Times ##########
    nclass = 20
    for cw in range( 1,3  ,1 ):
        
        c_weight = {0 : cw * 0.1 , 1 : 1}     
        ########## Set Training Strate ##########
        for t in range(0, nclass):
            
            ########### Shulffle All Training Data ##########
            pssm_pos, pssm_neg, oneofkey_pos, oneofkey_neg, physical_pos, physical_neg = shufflewrr(pssm_pos, pssm_neg, oneofkey_pos, oneofkey_neg, physical_pos, physical_neg)
            
            ########## A For Positive Data Number Set ##########
            a = int(len(oneofkey_pos)*0.8)
            
            ########## Oneofkey Training ##########
            train_oneofkey_pos = oneofkey_pos[0:a]
            train_oneofkey_neg = oneofkey_neg[0:a]
            
            ########## Physical Training ##########
            train_physical_pos = physical_pos[0:a]
            train_physical_neg = physical_neg[0:a]
            
            ########## Pssm Training ##########
            train_pssm_pos = pssm_pos[0:a]
            train_pssm_neg = pssm_neg[0:a]        
            
            print('total train',len(train_oneofkey_pos),len(train_oneofkey_neg),'blblblblbl',len(train_physical_pos),len(train_physical_neg),len(train_pssm_pos),len(train_pssm_neg))
            
            ########## Pos Concat Neg ##########
            train_oneofkey_all = pd.concat([train_oneofkey_pos,train_oneofkey_neg])
            train_physical_all = pd.concat([train_physical_pos,train_physical_neg])
            train_pssm_all = train_pssm_pos + train_pssm_neg
            ########## Shuffle Again ##########
            train_pssm_all,train_oneofkey_all,train_physical_all = shufflePosNeg(train_pssm_all,train_oneofkey_all,train_physical_all)        
            
            ########## Dprocess For Codes ##########
            train_oneofkey_all = pd.DataFrame(train_oneofkey_all)
            train_oneofkeyX,train_oneofkeyY = convertRawToXY(train_oneofkey_all.as_matrix(),codingMode=0)
            train_physical_all = pd.DataFrame(train_physical_all)
            train_physicalX,train_physicalY = convertRawToXY(train_physical_all.as_matrix(),codingMode=6)                
            train_pssmX = convertRawToXY(train_pssm_all,codingMode=7)
            train_pssmY = train_oneofkeyY     
            
            ########## Del Trainall ##########
            del train_oneofkey_all,train_physical_all,train_pssm_all
            
            ########## MultiCNN ##########
            if(t==0):            
                models = MultiCNN(train_oneofkeyX,train_oneofkeyY,train_physicalX,train_pssmX,
                                  pre_train_seq_path = 'bestmodel/best - oneofk - model.h5',
                                  pre_train_physical_path = 'bestmodel/best - physical - model.h5',
                                  pre_train_pssm_path = 'bestmodel/best - pssm - model.h5',                              
                                  nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,
                                  class_weights= c_weight,forkinas=forkinas,compiletimes=t)
                #predict_classes = kutils.probas_to_classes(models.predict([test_oneofkeyX_t,test_physicalX_t,testPssmX_t] ,batch_size=2048))
                #predict_classes = K.round(models.predict(test_physicalX))
                #print('sklearn mcc',sklearn.metrics.matthews_corrcoef(test_physicalY[:,1], predict_classes))     
                #print('our calculation',calculate_performance(len(test_physicalY), test_physicalY[:,1], predict_classes))            
                #print('No.'+ str(t)+':', models.metrics_names,models.evaluate([test_oneofkeyX_t,test_physicalX_t,testPssmX_t], test_oneofkeyY, batch_size=2048))            
    
            else:
                models = MultiCNN(train_oneofkeyX,train_oneofkeyY,train_physicalX,train_pssmX,
                                  pre_train_seq_path = 'bestmodel/best - oneofk - model.h5',
                                  pre_train_physical_path = 'bestmodel/best - physical - model.h5',
                                  pre_train_pssm_path = 'bestmodel/best - pssm - model.h5',                              
                                  nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,
                                  class_weights= c_weight,forkinas=forkinas,compiletimes=t,compilemodels=models)          
                #models.save('physicalfinal',overwrite=True)
                #models.save_weights('physicalweightfinal',overwrite=True)            
                #predict_classes = kutils.probas_to_classes(models.predict([test_oneofkeyX_t,test_physicalX_t,testPssmX_t] ,batch_size=2048))
                #predict_classes = K.round(models.predict(test_physicalX))
                #print('sklearn mcc',sklearn.metrics.matthews_corrcoef(test_physicalY[:,1], predict_classes))     
                #print('our calculation', calculate_performance(len(test_physicalY), test_physicalY[:,1], predict_classes))            
                #print('No.'+ str(t)+':', models.metrics_names,models.evaluate([test_oneofkeyX_t,test_physicalX_t,testPssmX_t], test_oneofkeyY, batch_size=2048))
                
            #predict testing set
            pred_proba = models.predict([test_oneofkeyX,test_physicalX,test_pssmX], batch_size=2048)
            predict_classes = kutils.probas_to_classes(pred_proba)
            #SAVE the prediction metrics
            with open('result/evaluation.txt', mode='a') as resFile:
                resFile.write(str(cw)+ ' ' +str(t) + ' '+calculate_performance(len(test_physicalY), test_physicalY[:,1], predict_classes,pred_proba[:,1])+ '\r\n')
            resFile.close()    
            true_label = test_oneofkeyY
            result = np.column_stack((true_label[:,1],pred_proba[:,1]))
            result = pd.DataFrame(result)
            result.to_csv(path_or_buf='result/result'+ '-' +str(t) +'-'+ str(cw) + '-'+'.txt',index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)         
    
    ########## Del Test Data ##########
    del test_pssm_pos,test_pssm_neg,test_oneofkey_pos,test_oneofkey_neg,test_physical_pos,test_physical_neg

def calculate_performance(test_num,labels,predict_y,predict_score):
    tp=0
    fp=0
    tn=0
    fn=0
    for index in range(test_num):
        if(labels[index]==1):
            if(labels[index] == predict_y[index]):
                tp += 1
            else:
                fn += 1
        else:
            if(labels[index] == predict_y[index]):
                tn += 1
            else:
                fp += 1
    acc = float(tp+tn)/test_num
    precision = float(tp)/(tp+fp+ sys.float_info.epsilon)
    sensitivity = float(tp)/(tp+fn+ sys.float_info.epsilon)
    specificity = float(tn)/(tn+fp+ sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    mcc = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    aps = average_precision_score(labels,predict_score)
    fpr,tpr,_ = roc_curve(labels,predict_score)
    aucResults = auc(fpr,tpr)
    
    
    
    strResults = 'tp '+ str(tp) + ' fn ' + str(fn) + ' tn ' + str(tn) + ' fp ' + str(fp)
    strResults = strResults + ' acc ' + str(acc) + ' precision ' + str(precision) + ' sensitivity ' + str(sensitivity)
    strResults = strResults + ' specificity ' + str(specificity) + ' f1 ' + str(f1) + ' mcc ' + str(mcc)
    strResults = strResults + ' aps ' + str(aps) + ' auc ' + str(aucResults)
    return strResults

def shufflewrr(data1_pos,data1_neg,data2_pos,data2_neg,data3_pos,data3_neg):
    
    index = [i for i in range(len(data1_pos))]
    random.shuffle(index)
    data2_pos = pd.DataFrame(data2_pos)
    data2_pos = data2_pos.as_matrix()[index]
    data2_pos_ss = pd.DataFrame(data2_pos)
    data3_pos = pd.DataFrame(data3_pos)    
    data3_pos = data3_pos.as_matrix()[index]
    data3_pos_ss = pd.DataFrame(data3_pos)
    data1_pos_ss=[]        
    for i in range(len(index)):
        data1_pos_ss.append(data1_pos[index[i]])
    
    index = [i for i in range(len(data1_neg))]
    random.shuffle(index)
    data2_neg =pd.DataFrame(data2_neg)
    data2_neg = data2_neg.as_matrix()[index]
    data2_neg_ss = pd.DataFrame(data2_neg)
    data3_neg =pd.DataFrame(data3_neg)
    data3_neg = data3_neg.as_matrix()[index]
    data3_neg_ss = pd.DataFrame(data3_neg)    
    data1_neg_ss=[]
    for i in range(len(index)):
        data1_neg_ss.append(data1_neg[index[i]]) 
        
    return data1_pos_ss,data1_neg_ss,data2_pos_ss,data2_neg_ss,data3_pos_ss,data3_neg_ss

def shufflePosNeg(data1,data2,data3):
    
    data1_over=[]
    index = [i for i in range(len(data1))]
    random.shuffle(index)
    data2_over = data2.as_matrix()[index]
    data2_over = pd.DataFrame(data2_over)
    data3_over = data3.as_matrix()[index]
    data3_over = pd.DataFrame(data3_over)
    
    for i in range(len(index)):
        data1_over.append(data1[index[i]])
    
    return data1_over,data2_over,data3_over

def precision_recall(true_label,pred_proba):
    lw = 2
    precision, recall, _ = precision_recall_curve(true_label,pred_proba)
    average_precision = average_precision_score(true_label, pred_proba)
    for i in range(len(precision)):
        print(precision[i], recall[i])
    plt.clf()
    plt.plot(recall, precision, color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
             ''.format(average_precision))  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to binary-class')
    plt.legend(loc="lower right")
    plt.show()    
    
def roc(true_label,pred_proba):
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
if __name__ == "__main__":
    '''
    with open('C:/Users/zhaoxy/Desktop/latestoneofk/nubisite_CNN2/result/result.txt','r') as thred:
        true_label = []
        pred_proba = []
        for line in thred:
            line1=line.split('\t')
            if(line1[0]=='1.0'):
                true_label.append(1)
            else:
                true_label.append(0)
            pred_proba.append(float(line1[1]))
    roc(true_label,pred_proba)
    precision_recall(true_label,pred_proba)
    '''
    main()         
   