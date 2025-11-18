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

    """
    # Example (for local testing)
    train = pd.read_csv("/tmp/gr-GnegativeGO/Global/Split-1/GnegativeGO-Split-Tr-1.csv")
    test = pd.read_csv("/tmp/gr-GnegativeGO/Global/Split-1/GnegativeGO-Split-Ts-1.csv")
    valid = pd.read_csv("/tmp/gr-GnegativeGO/Global/Split-1/GnegativeGO-Split-Vl-1.csv")
    start = 1717
    directory = "/tmp/gr-GnegativeGO/Global/Split-1"
    fold = 1
    """

    print("\n%==============================================%")
    #print("train: ", sys.argv[1])
    #print("valid: ", sys.argv[2])
    #print("test: ", sys.argv[3])
    #print("start: ", sys.argv[4])
    print("directory: ", sys.argv[5])
    #print("fold: ", sys.argv[6])
    print("%==============================================%\n")
     
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

        
    # ======= TREINO =======
    start_train_time = time.time()
    rf.fit(X_train, Y_train)
    end_train_time = time.time()
    training = end_train_time - start_train_time


    # ======= PREDIÇÃO BINÁRIA =======
    start_test_time = time.time()
    binary_predictions = rf.predict(X_test)
    binary_df = pd.DataFrame(binary_predictions, columns=labels_y_test)
    end_test_time = time.time()
    testing_bin = end_test_time - start_test_time


    # ======= PREDIÇÃO DE PROBABILIDADES =======
    start_test_time = time.time()        
    probas_list = rf.predict_proba(X_test)
    end_test_time = time.time()
    testing_proba = end_test_time - start_test_time


    # ======= PEGANDO APENAS A PROBABILIDADE DE PERTENCER =======        
    # empilha as colunas de probabilidade de pertencer (classe 1)
    probabilities = np.array([p[:, 1] for p in probas_list]).T    
    # converte em dataframe
    probabilities_df = pd.DataFrame(probabilities, columns=labels_y_test)


    # ======= CONVERTENDO TODAS AS PREDIÇÕES =======    
    columns = []
    for cls in labels_y_test:
        columns.extend([f"{cls}_0", f"{cls}_1"])
    
    # Empilhando os arrays para ficar 2 colunas por classe
    probas_arrays = [np.hstack([p[:, 0].reshape(-1,1), p[:, 1].reshape(-1,1)]) for p in probas_list]

    # Concatenando todas as colunas
    all_probas = np.hstack(probas_arrays)

    # monta o dataframe
    probas_df = pd.DataFrame(all_probas, columns=columns)



    # ======= SALVANDO OS CSVS =======        
    true = os.path.join(directory, "y_true.csv")
    binary = os.path.join(directory, "y_pred_bin.csv")
    proba = os.path.join(directory, "y_pred_proba.csv")
    original = os.path.join(directory, "y_proba_original.csv")
    test[labels_y_test].to_csv(true, index=False)    
    probabilities_df.to_csv(proba, index=False)
    binary_df.to_csv(binary, index=False)
    probas_df.to_csv(original, index=False)


    # ======= SAVE TIME =======    
    df_timing = pd.DataFrame([[        
        training,
        testing_bin,
        testing_proba
    ]], columns=["training", "testing_bin", "testing_proba"])
    df_timing.to_csv(os.path.join(directory, "runtime-python.csv"), index=False)
    # df_timing


    # =========== SAVE MEASURES ===========   
    #metrics_df = eval.multilabel_curves_measures(Y_test, pd.DataFrame(probabilities, columns=labels_y_test))
    #metrics_df.to_csv(os.path.join(directory, "results-python.csv"), index=False)        
    
    # metrics_df, ignored_df = eval.multilabel_curve_metrics(Y_test, probabilities_df)        
    # name = (directory + "/results-python.csv") 
    # metrics_df.to_csv(name, index=False)      
    # name = (directory + "/ignored-classes.csv") 
    # ignored_df.to_csv(name, index=False)    
       

    # ======= SAVE MODEL SIZE =======
    model_buffer = io.BytesIO()
    pickle.dump(rf, model_buffer)
    model_size_bytes = model_buffer.tell()
    pd.DataFrame({'size': [model_size_bytes]}).to_csv(
        os.path.join(directory, "model-size.csv"), index=False
    )
