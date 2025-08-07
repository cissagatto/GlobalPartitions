##############################################################################
# GLOBAL PARTITIONS MULTI-LABEL CLASSIFICATION                               #
# Copyright (C) 2025                                                         #
#                                                                            #
# This code is free software: you can redistribute it and/or modify it under #
# the terms of the GNU General Public License as published by the Free       #
# Software Foundation, either version 3 of the License, or (at your option)  #
# any later version. This code is distributed in the hope that it will be    #
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General   #
# Public License for more details.                                           #
#                                                                            #
# 1 - Prof PhD Elaine Cecilia Gatto                                          #
# 2 - Prof PhD Ricardo Cerri                                                 #
# 3 - Prof PhD Mauri Ferrandin                                               #
# 4 - Prof PhD Celine Vens                                                   #
# 5 - PhD Felipe Nakano Kenji                                                #
# 6 - Prof PhD Jesse Read                                                    #
#                                                                            #
# 1 = Federal University of São Carlos - UFSCar - https://www2.ufscar.br     #
# Campus São Carlos | Computer Department - DC - https://site.dc.ufscar.br | #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br | Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       # 
#                                                                            # 
# 1 = Federal University of Lavras - UFLA                                    #
#                                                                            # 
# 2 = State University of São Paulo - USP                                    #
#                                                                            # 
# 3 - Federal University of Santa Catarina Campus Blumenau - UFSC            #
# https://ufsc.br/                                                           #
#                                                                            #
# 4 and 5 - Katholieke Universiteit Leuven Campus Kulak Kortrijk Belgium     #
# Medicine Department - https://kulak.kuleuven.be/                           #
# https://kulak.kuleuven.be/nl/over_kulak/faculteiten/geneeskunde            #
#                                                                            #
# 6 - Ecole Polytechnique | Institut Polytechnique de Paris | 1 rue Honoré   #
# d’Estienne d’Orves - 91120 - Palaiseau - FRANCE                            #
#                                                                            #
##############################################################################


import sys
import platform
import os
import io

#FolderRoot = os.path.expanduser('/lapix/arquivos/elaine/GlobalPartitions/Python')
#os.chdir(FolderRoot)
#current_directory = os.getcwd()
#sys.path.append('..')

import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
import importlib
import evaluation as eval
importlib.reload(eval)
import measures as ms
importlib.reload(ms)


if __name__ == '__main__':   
    
    # Getting command-line arguments
    train = pd.read_csv(sys.argv[1])    # training set
    valid = pd.read_csv(sys.argv[2])    # validation set
    test = pd.read_csv(sys.argv[3])     # test set
    start = int(sys.argv[4])            # label start index
    directory = sys.argv[5]             # directory to save predictions
    fold = sys.argv[6]             # directory to save predictions

    # Example (for local testing)
    # train = pd.read_csv("/tmp/gr-emotions/Global/Split-1/emotions-Split-Tr-1.csv")
    # test = pd.read_csv("/tmp/gr-emotions/Global/Split-1/emotions-Split-Ts-1.csv")
    # valid = pd.read_csv("/tmp/gr-emotions/Global/Split-1/emotions-Split-Vl-1.csv")
    # start = 72
    # directory = "/tmp/gr-emotions/Global/Split-1"
    # fold = 1

    print("\n\n%==============================================%")
    #print("label train: ", sys.argv[1])
    #print("label valid: ", sys.argv[2])
    #print("label test: ", sys.argv[3])
    #print("label start: ", sys.argv[4])
    #print("directory: ", sys.argv[5])
    print("fold: ", sys.argv[6])
    print("%==============================================%\n\n")
     
    # Merge train and validation sets
    train = pd.concat([train, valid], axis=0).reset_index(drop=True)
    
    # Split features and labels for training set
    X_train = train.iloc[:, :start]    
    Y_train = train.iloc[:, start:]    
    
    # Split features and labels for test set
    X_test = test.iloc[:, :start]    
    Y_test = test.iloc[:, start:]    
    
    # Get label and attribute names
    labels_y_train = list(Y_train.columns)
    labels_y_test = list(Y_test.columns)    
    attr_x_train = list(X_train.columns)
    attr_x_test = list(X_test.columns)
    
    # Classifier parameters
    random_state = 1234    
    n_estimators = 200    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    start_train_time = time.time() # Measure training time
    rf.fit(X_train, Y_train) # Train model
    end_train_time = time.time()
    training_time = end_train_time - start_train_time  
    
    # setando nome do diretorio e arquivo para salvar
    true = (directory + "/y_true.csv")     
    pred = (directory + "/y_pred_bin.csv") 
    proba = (directory + "/y_pred_proba.csv")  
    proba_original = (directory + "/proba_original.csv")  

    # predições probabilísticas
    start_time_test_proba = time.time()
    probabilities = eval.safe_predict_proba(rf, X_test, Y_train)
    end_time_test_proba = time.time()
    test_duration_proba = end_time_test_proba - start_time_test_proba
    probabilities.to_csv(proba, index=False)

    Y_test.to_csv(true, index=False)

    times_df = pd.DataFrame({
        'train_duration': [training_time],
        'test_duration_proba': [test_duration_proba],
        #'test_duration_bin': [test_duration_bin]
    })
    times_path = os.path.join(directory, "runtime-python.csv")
    times_df.to_csv(times_path, index=False)
    

    # =========== SAVE MEASURES ===========   
    metrics_df, ignored_df = eval.multilabel_curve_metrics(Y_test, probabilities)    
    name = (directory + "/results-python.csv") 
    metrics_df.to_csv(name, index=False)  
    name = (directory + "/ignored-classes.csv") 
    ignored_df.to_csv(name, index=False)  
     

    # =========== SAVE MODEL SIZE EM BYTES ===========
    model_buffer = io.BytesIO()
    pickle.dump(rf, model_buffer)
    model_size_bytes = model_buffer.tell()
    model_size_df = pd.DataFrame({
        'model_size_bytes': [model_size_bytes]
    })
    model_size_df.to_csv(os.path.join(directory, "model-size.csv"), index=False)
