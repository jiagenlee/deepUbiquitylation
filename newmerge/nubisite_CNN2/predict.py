import sys
import pandas as pd
import numpy as np
import argparse
import csv

def main():
    parser=argparse.ArgumentParser(description='MusiteDeep prediction tool for general, kinase-specific phosphorylation prediction or custom PTM prediction by using custom models.')
    parser.add_argument('-input',  dest='inputfile', type=str, help='Protein sequences to be predicted in fasta format.', required=True)
    parser.add_argument('-predict-type',  
                        dest='predicttype', 
                        type=str, 
                        help='predict types. \'general\' for general human phosphorylation prediction by models pre-trained in MusiteDeep. \n \
                        \'kinase\' for kinase-specific human phosphorylation prediction by models pre-trained in MusiteDeep.\n \
                        \'custom\' for custom PTM prediction. a custom model must be provided by -model-prefix. \
                        It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.', required=True)
    parser.add_argument('-output',  dest='outputfile', type=str, help='prefix of the prediction results.', required=True)
    parser.add_argument('-kinase',  dest='kinase', type=str, help='if -predict-type is \'kinase\', -kinase indicates the specific kinase, currently we accept \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\'.', required=False,default=None)
    parser.add_argument('-model-prefix',  dest='modelprefix', type=str, help='prefix of custom model used for prediciton. If donnot have one, please run train_general.py to train a custom general PTM model or run train_kinase.py to train a custom kinase-specific PTM model.', required=False,default=None)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that to be predicted, only used when -predict-type is \'general\'. For multiple residues, seperate each with \',\'',required=False,default="S,T,Y")
    
    args = parser.parse_args()
    
    kinaselist=["CDK","PKA","CK2","MAPK","PKC"];
    
    inputfile=args.inputfile;
    outputfile=args.outputfile;
    predicttype=args.predicttype;
    residues=args.residues.split(",")
    kinase=args.kinase;
    modelprefix=args.modelprefix;
    
    
    if predicttype == 'general': #prediction for general phosphorylation
        from methods.DProcess import convertRawToXY
        from methods.EXtractfragment_sort import extractFragforPredict
        from methods.multiCNN import MultiCNN
        nclass=5
        window=16
        results_ST=None
        results_Y=None
        #################for S and T
        if("Y" in residues):
          residues.remove("Y")
        
        if("S" in residues or "T" in residues):
               print "General phosphorylation prediction for S or T: \n"        
               testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=residues)
               testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
               predictproba=np.zeros((testX.shape[0],2))
               models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
               model="./models/models_ST_HDF5model_class"
               for bt in range(nclass):
                   models.load_weights(model+str(bt))
                   predictproba+=models.predict(testX)
                   print("Done predicting by model of class "+str(bt)+"\n");
               
               predictproba=predictproba/nclass;
               poses=poses+1;
               results_ST=np.column_stack((ids,poses,focuses,predictproba[:,1]))
               result=pd.DataFrame(results_ST)
               result.to_csv(outputfile+"_general_phosphorylation_SorT.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
              
        #########for Y################
        residues=args.residues.split(",")
        if("Y" in residues):
           print "General phosphorylation prediction for Y: \n"        
           testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=("Y"))
           testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
           predictproba=np.zeros((testX.shape[0],2))
           models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
           model="./models/models_Y_HDF5model_"
           nclass_init=5;
           nclass=3;
           for ini in range(nclass_init):
                   for bt in range(nclass):
                           models.load_weights(model+'ini'+str(ini)+'_class'+str(bt))
                           predictproba+=models.predict(testX)
                           print("Done predicting by model of class "+str(bt)+" and initial class "+str(ini)+" \n");
           
           predictproba=predictproba/(nclass*nclass_init);
           poses=poses+1;
           results_Y=np.column_stack((ids,poses,focuses,predictproba[:,1]))
           result=pd.DataFrame(results_Y)
           result.to_csv(outputfile+"_general_phosphorylation_Y.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
        
        print "Successfully predicted for general phosphorylation !\n";
    elif predicttype == 'kinase':
         if kinase is None or kinase not in kinaselist:
            print "wrong parameter for -kinase! Must be one of \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\' !\n";
            exit()
         else: #prediction for kinas
               from methods.DProcess import convertRawToXY
               from methods.EXtractfragment_sort import extractFragforPredict
               from methods.multiCNN import MultiCNN
               print "Kinase-specific prediction for "+str(kinase)+" !\n";
               nclass_init=5
               nclass=3
               window=16
               testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=("S","T"))
               testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
               predictproba=np.zeros((testX.shape[0],2))
               models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
               model="./models/"+str(kinase)+"_model_"
               for ini in range(nclass_init):
                   for bt in range(nclass):
                           models.load_weights(model+'ini'+str(ini)+'_class'+str(bt))
                           predictproba+=models.predict(testX)
                           print("Done predicting by model of class "+str(bt)+" and initial class "+str(ini)+" \n");
               
               predictproba=predictproba/(nclass*nclass_init);
               poses=poses+1;
               results=np.column_stack((ids,poses,focuses,predictproba[:,1]))
               result=pd.DataFrame(results)
               result.to_csv(outputfile+"_"+str(kinase)+".txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
               print "Successfully predicted for "+str(kinase)+" !\n";
         
    elif predicttype == 'custom':
        if modelprefix is None:
           print "If you want to do prediction by a custom model, please specify the prefix for an existing custom model by -model-prefix!\n\
           It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
           If you don't have such files, please run train_general.py or train_kinase.py to get the custom model first!\n"
           exit()
        else: #custom prediction
          model=modelprefix+str("_HDF5model")
          parameter=modelprefix+str("_parameters")
          try:
              f=open(parameter,'r')
          except IOError:
              print 'cannot open '+ parameter+" ! check if the model exists. please run train_general.py or train_kinase.py to get the custom model first!\n"
          else:
               f= open(parameter, 'r')
               parameters=f.read()
               f.close()
          from methods.DProcess import convertRawToXY
          from methods.EXtractfragment_sort import extractFragforPredict
          from methods.multiCNN import MultiCNN
          nclass=int(parameters.split("\t")[0])
          window=int(parameters.split("\t")[1])
          residues=parameters.split("\t")[2]
          residues=residues.split(",")
          testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=residues)
          
          testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
          predictproba=np.zeros((testX.shape[0],2))
          models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
          
          if(parameters.split("\t")[3]=="kinase-specific"):
               nclass_ini=int(parameters.split("\t")[4])
               for ini in range(nclass_ini):
                   for bt in range(nclass):
                        models.load_weights(model+'_ini'+str(ini)+'_class'+str(bt))
                        predictproba+=models.predict(testX)
               
          else:
             nclass_ini=1;
             for bt in range(nclass):
                 models.load_weights(model+"_class"+str(bt))
                 predictproba+=models.predict(testX)
          
          predictproba=predictproba/(nclass*nclass_ini);
          poses=poses+1;
          results=np.column_stack((ids,poses,focuses,predictproba[:,1]))
          result=pd.DataFrame(results)
          result.to_csv(outputfile+"_custom.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
          print "Successfully predicted from custom models !\n";
          
    else: 
       print "wrong parameter for -predict-type!\n";
       exit();
    
     

    
if __name__ == "__main__":
    main()         
   