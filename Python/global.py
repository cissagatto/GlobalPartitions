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
    # train = pd.read_csv("/home/cissagatto/Global/data/emotions-Split-Tr-1.csv")
    # test = pd.read_csv("/home/cissagatto/Global/data/emotions-Split-Ts-1.csv")
    # valid = pd.read_csv("/home/cissagatto/Global/data/emotions-Split-Vl-1.csv")
    # start = 72
    # directory = "/home/cissagatto/Global"

    print("\n\n%==============================================%")
    print("label train: ", sys.argv[1])
    print("label valid: ", sys.argv[2])
    print("label test: ", sys.argv[3])
    print("label start: ", sys.argv[4])
    print("directory: ", sys.argv[5])
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

    # Initialize classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    start_train_time = time.time() # Measure training time
    rf.fit(X_train, Y_train) # Train model
    end_train_time = time.time()
    training_time = end_train_time - start_train_time  
    
    start_test_time = time.time() # Measure prediction time (binary predictions)
    y_pred_bin = pd.DataFrame(rf.predict(X_test)) # Binary predictions
    y_pred_bin.columns = labels_y_test
    end_test_time = time.time()
    testing_time_bin = end_test_time - start_test_time

    start_test_time = time.time() # Measure prediction time (binary predictions)
    y_pred_proba = rf.predict_proba(X_test) # Probabilistic predictions
    end_test_time = time.time()
    testing_time_proba = end_test_time - start_test_time

    # Prepare dataframe
    timing_data = [
        ["training_time", training_time],
        ["testing_time_bin", testing_time_bin],
        ["testing_time_proba", testing_time_proba]
    ]

    df_timing = pd.DataFrame(timing_data, columns=["Process", "Time (s)"])

    # Save to CSV
    name_csv = os.path.join(directory, "runtime-python.csv")
    df_timing.to_csv(name_csv, index=False)   
    

    # Measure pickle size in memory
    buffer_pickle = io.BytesIO()
    pickle.dump(rf, buffer_pickle)
    size_pickle_bytes = buffer_pickle.tell()

    # Measure joblib size in memory
    buffer_joblib = io.BytesIO()
    joblib.dump(rf, buffer_joblib)
    size_joblib_bytes = buffer_joblib.tell()

    # Prepare dataframe with only bytes
    model_sizes = [
        ["pickle", size_pickle_bytes],
        ["joblib", size_joblib_bytes]
    ]

    df_sizes = pd.DataFrame(model_sizes, columns=["Format", "Size (Bytes)"])

    # Save to CSV
    name_csv = os.path.join(directory, "model-sizes.csv")
    df_sizes.to_csv(name_csv, index=False)
    
    # Set output file paths
    name_true = os.path.join(directory, "y_true.csv")
    name_pred_bin = os.path.join(directory, "y_pred_bin.csv")    
    name_pred_proba = os.path.join(directory, "y_pred_proba.csv")
    name_pred_proba_original = os.path.join(directory, "proba_original.csv")
    
    # Save true labels and binary predictions
    y_pred_bin.to_csv(name_pred_bin, index=False)
    Y_test.to_csv(name_true, index=False)    
    
    # Save probabilistic predictions
    ldf1 = []
    for n in range(0, len(y_pred_proba)):
        res = y_pred_proba[n]
        res1 = pd.DataFrame(res)
        res1.columns = [f'prob_{n}_0', f'prob_{n}_1']
        ldf1.append(res1)      
    
    final = pd.concat(ldf1, axis=1)
    final.to_csv(name_pred_proba_original, index=False)    

    # Seleciona apenas as colunas cujo nome termina com '_1'
    final_1 = final.loc[:, final.columns.str.endswith('_1')]
    final_1.columns = labels_y_test
    final_1.to_csv(name_pred_proba, index=False)    

    print("\nCOMPUTE CURVES")
    res_curves = eval.multilabel_curves_measures(Y_test, final_1)    
    name = (directory + "/results-python.csv") 
    res_curves.to_csv(name, index=False)