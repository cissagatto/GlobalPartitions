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




###############################################################################
# SET WORKSAPCE                                                               #
###############################################################################
#library(here)
#library(stringr)
#FolderRoot <- here::here()
#FolderScripts <- here::here("R")




###############################################################################
#
###############################################################################
converteArff <- function(arg1, arg2, arg3, FolderUtils){  
  str = paste("java -jar ", FolderUtils,  "/R_csv_2_arff.jar ", 
              arg1, " ", arg2, " ", arg3, sep="")
  print(system(str))
  cat("\n\n")  
}


###############################################################################
#
###############################################################################
properties.datasets <- function(parameters){
  
  fold = c(0)
  num.attributes = c(0)
  num.instances = c(0)
  num.inputs = c(0)
  num.labels = c(0)
  num.labelsets = c(0)
  num.single.labelsets = c(0)
  max.frequency = c(0)
  cardinality = c(0)
  density = c(0)
  meanIR = c(0)
  scumble = c(0)
  scumble.cv = c(0)
  tcs = c(0)
  
  measures.treino = data.frame(fold, num.attributes, num.instances, num.inputs,
                               num.labels, num.labelsets, num.single.labelsets,
                               max.frequency, cardinality, density, meanIR,
                               scumble, scumble.cv, tcs)
  
  measures.teste = data.frame(fold, num.attributes, num.instances, num.inputs,
                              num.labels, num.labelsets, num.single.labelsets,
                              max.frequency, cardinality, density, meanIR,
                              scumble, scumble.cv, tcs)
  
  measures.val = data.frame(fold, num.attributes, num.instances, num.inputs,
                            num.labels, num.labelsets, num.single.labelsets,
                            max.frequency, cardinality, density, meanIR,
                            scumble, scumble.cv, tcs)
  
  measures.tv = data.frame(fold, num.attributes, num.instances, num.inputs,
                            num.labels, num.labelsets, num.single.labelsets,
                            max.frequency, cardinality, density, meanIR,
                            scumble, scumble.cv, tcs)
  
  folder = paste(parameters$Directories$FolderGlobal, 
                 "/", parameters$Dataset.Name,
                 "/Properties", sep="")
  if(dir.exists(folder)==FALSE){dir.create(folder)}
  
  
  f = 1
  while(f<=parameters$Config.File$Number.Folds){
    
    cat("\nFold ", f)
    
    ####################################################################
    folderSave = paste(folder, "/Split-", f, sep="")
    if(dir.exists(folderSave)==FALSE){dir.create(folderSave)}
    
    
    ####################################################################
    nome = paste(parameters$Directories$FolderNamesLabels , 
                 "/", parameters$Config.File$Dataset.Name,
                 "-NamesLabels.csv", sep="")
    rotulos = data.frame(read.csv(nome))
    names(rotulos) = c("index", "names.labels")
    
    
    ####################################################################
    nome = paste(parameters$Directories$FolderCVTR, 
                 "/", parameters$Config.File$Dataset.Name,
                 "-Split-Tr-", f, ".csv", sep="")
    treino = data.frame(read.csv(nome))
    
    nome = paste(parameters$Directories$FolderCVTS, 
                 "/", parameters$Config.File$Dataset.Name,
                 "-Split-Ts-", f, ".csv", sep="")
    teste = data.frame(read.csv(nome))
    
    nome = paste(parameters$Directories$FolderCVVL, 
                 "/", parameters$Config.File$Dataset.Name,
                 "-Split-Vl-", f, ".csv", sep="")
    val = data.frame(read.csv(nome))
    
    tv = rbind(treino, val)
    
    
    ##################################################################
    treino.labels = treino[,parameters$Dataset.Info$LabelStart:parameters$Dataset.Info$LabelEnd]
    teste.labels = teste[,parameters$Dataset.Info$LabelStart:parameters$Dataset.Info$LabelEnd]
    val.labels = val[,parameters$Dataset.Info$LabelStart:parameters$Dataset.Info$LabelEnd]
    tv.labels = tv[,parameters$Dataset.Info$LabelStart:parameters$Dataset.Info$LabelEnd]
    
    
    ##########################################################################
    treino.sd = apply(treino.labels , 2, sd)
    treino.mean = apply(treino.labels , 2, mean)
    treino.median = apply(treino.labels , 2, median)
    treino.sum = apply(treino.labels , 2, sum)
    treino.max = apply(treino.labels , 2, max)
    treino.min = apply(treino.labels , 2, min)
    treino.quartis = apply(treino.labels, 2, quantile, 
                           probs = c(0.10, 0.25, 0.50, 0.75, 0.90))
    treino.summary = rbind(sd = treino.sd, mean = treino.mean, 
                           median = treino.median,
                           sum = treino.sum, max = treino.max, 
                           min = treino.min, treino.quartis)
    name = paste(folderSave, "/summary-train-", f, ".csv", sep="")
    write.csv(treino.summary, name)
    
    
    ##########################################################################
    teste.sd = apply(teste.labels , 2, sd)
    teste.mean = apply(teste.labels , 2, mean)
    teste.median = apply(teste.labels , 2, median)
    teste.sum = apply(teste.labels , 2, sum)
    teste.max = apply(teste.labels , 2, max)
    teste.min = apply(teste.labels , 2, min)
    teste.quartis = apply(teste.labels, 2, quantile,
                          probs = c(0.10, 0.25, 0.50, 0.75, 0.90))
    teste.summary = rbind(sd = teste.sd, mean = teste.mean, 
                          median = teste.median,
                          sum = teste.sum, max = teste.max, 
                          min = teste.min, teste.quartis)
    name = paste(folderSave, "/summary-test-", f, ".csv", sep="")
    write.csv(teste.summary, name)
    
    
    ##########################################################################
    val.sd = apply(val.labels , 2, sd)
    val.mean = apply(val.labels , 2, mean)
    val.median = apply(val.labels , 2, median)
    val.sum = apply(val.labels , 2, sum)
    val.max = apply(val.labels , 2, max)
    val.min = apply(val.labels , 2, min)
    val.quartis = apply(val.labels, 2, quantile, 
                       probs = c(0.10, 0.25, 0.50, 0.75, 0.90))
    val.summary = rbind(sd = val.sd, mean = val.mean, 
                        median = val.median,
                        sum = val.sum, max = val.max, 
                        min = val.min, val.quartis)
    name = paste(folderSave, "/summary-val-", f, ".csv", sep="")
    write.csv(val.summary, name)
    
    
    ##########################################################################
    tv.sd = apply(tv.labels , 2, sd)
    tv.mean = apply(tv.labels , 2, mean)
    tv.median = apply(tv.labels , 2, median)
    tv.sum = apply(tv.labels , 2, sum)
    tv.max = apply(tv.labels , 2, max)
    tv.min = apply(tv.labels , 2, min)
    tv.quartis = apply(tv.labels, 2, quantile, 
                        probs = c(0.10, 0.25, 0.50, 0.75, 0.90))
    tv.summary = rbind(sd = tv.sd, mean = tv.mean, 
                        median = tv.median,
                        sum = tv.sum, max = tv.max, 
                        min = tv.min, tv.quartis)
    name = paste(folderSave, "/summary-tv-", f, ".csv", sep="")
    write.csv(tv.summary, name)
    
    
    ##################################################################
    treino.num.positive.instances = apply(treino.labels , 2, sum)
    teste.num.positive.instances = apply(teste.labels , 2, sum)
    val.num.positive.instances = apply(val.labels , 2, sum)
    tv.num.positive.instances = apply(tv.labels , 2, sum)
    
    
    ##################################################################
    treino.num.instancias = nrow(treino)
    treino.num.negative.instances = treino.num.instancias - treino.num.positive.instances 
    
    teste.num.instancias = nrow(teste)
    teste.num.negative.instances = teste.num.instancias - teste.num.positive.instances 
    
    val.num.instancias = nrow(val)
    val.num.negative.instances = val.num.instancias - val.num.positive.instances 
    
    tv.num.instancias = nrow(tv)
    tv.num.negative.instances = tv.num.instancias - treino.num.positive.instances 
    
    todos = rbind(treino.num.positive.instances, treino.num.negative.instances,
          teste.num.positive.instances, teste.num.negative.instances,
          val.num.positive.instances, val.num.negative.instances,
          tv.num.positive.instances, tv.num.negative.instances)
    
    name = paste(folderSave, "/instances-pos-neg-", f, ".csv", sep="")
    write.csv(todos, name)
    
    
    ##########################################################################
    treino.num.positive.instances = data.frame(treino.num.positive.instances)
    treino.num.negative.instances = data.frame(treino.num.negative.instances)
    
    teste.num.positive.instances = data.frame(teste.num.positive.instances)
    teste.num.negative.instances = data.frame(teste.num.negative.instances)
    
    val.num.positive.instances = data.frame(val.num.positive.instances)
    val.num.negative.instances = data.frame(val.num.negative.instances)
    
    tv.num.positive.instances = data.frame(tv.num.positive.instances)
    tv.num.negative.instances = data.frame(tv.num.negative.instances)
    
    ##################################################################
    label = rownames(treino.num.positive.instances)
    
    ##################################################################
    treino.num.positive.instances = data.frame(label , frequency = treino.num.positive.instances$treino.num.positive.instances)
    treino.num.negative.instances = data.frame(label , frequency = treino.num.negative.instances$treino.num.negative.instances)
    
    teste.num.positive.instances = data.frame(label , frequency = teste.num.positive.instances$teste.num.positive.instances)
    teste.num.negative.instances = data.frame(label , frequency = teste.num.negative.instances$teste.num.negative.instances)
    
    val.num.positive.instances = data.frame(label , frequency = val.num.positive.instances$val.num.positive.instances)
    val.num.negative.instances = data.frame(label , frequency = val.num.negative.instances$val.num.negative.instances)
    
    tv.num.positive.instances = data.frame(label , frequency = tv.num.positive.instances$tv.num.positive.instances)
    tv.num.negative.instances = data.frame(label , frequency = tv.num.negative.instances$tv.num.negative.instances)
    
    ##########################################################################
    treino.num.positive.instances = arrange(treino.num.positive.instances, desc(frequency))
    ultimo = nrow(treino.num.positive.instances)
    treino.max = data.frame(treino.num.positive.instances[1,])
    treino.min = data.frame(treino.num.positive.instances[ultimo,])
    
    teste.num.positive.instances = arrange(teste.num.positive.instances, desc(frequency))
    ultimo = nrow(teste.num.positive.instances)
    teste.max = data.frame(teste.num.positive.instances[1,])
    teste.min = data.frame(teste.num.positive.instances[ultimo,])
    
    val.num.positive.instances = arrange(val.num.positive.instances, desc(frequency))
    ultimo = nrow(val.num.positive.instances)
    val.max = data.frame(val.num.positive.instances[1,])
    val.min = data.frame(val.num.positive.instances[ultimo,])
    
    tv.num.positive.instances = arrange(tv.num.positive.instances, desc(frequency))
    ultimo = nrow(tv.num.positive.instances)
    tv.max = data.frame(tv.num.positive.instances[1,])
    tv.min = data.frame(tv.num.positive.instances[ultimo,])
    
    max.min = rbind(treino.max, treino.min,
                    teste.max, teste.min,
                    val.max, val.min,
                    tv.max, tv.min)
    
    set = c("train.max", "train.min",
             "teste.max", "teste.min",
             "val.max", "val.min",
             "tv.max", "tv.min")
    
    final = data.frame(set, max.min)
            
    name = paste(folderSave, "/labels-max-min-", f, ".csv", sep="")
    write.csv(final, name, row.names = FALSE)
    

    ##########################################################################
    labels.indices = seq(parameters$Dataset.Info$LabelStart, parameters$Dataset.Info$LabelEnd, by=1)
    mldr.treino = mldr_from_dataframe(treino, labelIndices = labels.indices)
    mldr.teste = mldr_from_dataframe(teste, labelIndices = labels.indices)
    mldr.val = mldr_from_dataframe(val, labelIndices = labels.indices)
    mldr.tv = mldr_from_dataframe(tv, labelIndices = labels.indices)
    
    
    ##########################################################################
    labelsets = data.frame(mldr.treino$labelsets)
    names(labelsets) = c("labelset", "frequency")
    name = paste(folderSave, "/labelsets-train-", f, ".csv", sep="")
    write.csv(labelsets, name, row.names = FALSE)
    
    rm(labelsets)
    labelsets = data.frame(mldr.teste$labelsets)
    names(labelsets) = c("labelset", "frequency")
    name = paste(folderSave, "/labelsets-test-", f, ".csv", sep="")
    write.csv(labelsets, name, row.names = FALSE)
    
    rm(labelsets)
    labelsets = data.frame(mldr.val$labelsets)
    names(labelsets) = c("labelset", "frequency")
    name = paste(folderSave, "/labelsets-val-", f, ".csv", sep="")
    write.csv(labelsets, name, row.names = FALSE)
    
    rm(labelsets)
    labelsets = data.frame(mldr.tv$labelsets)
    names(labelsets) = c("labelset", "frequency")
    name = paste(folderSave, "/labelsets-tv-", f, ".csv", sep="")
    write.csv(labelsets, name, row.names = FALSE)
    
    
    ##########################################################################
    labels = data.frame(mldr.treino$labels)
    name = paste(folderSave, "/labels-train-", f, ".csv", sep="")
    write.csv(labels, name)
    
    rm(labels)
    labels = data.frame(mldr.teste$labels)
    name = paste(folderSave, "/labels-test-", f, ".csv", sep="")
    write.csv(labels, name)
    
    rm(labels)
    labels = data.frame(mldr.val$labels)
    name = paste(folderSave, "/labels-val-", f, ".csv", sep="")
    write.csv(labels, name)
    
    rm(labels)
    labels = data.frame(mldr.tv$labels)
    name = paste(folderSave, "/labels-tv-", f, ".csv", sep="")
    write.csv(labels, name)
    
    
    ##########################################################################  
    properties = data.frame(mldr.treino$measures)
    properties = cbind(fold = f, properties)
    measures.treino = rbind(measures.treino, properties)
    #name = paste(folderSave , "/properties-train-", f, ".csv", sep="")
    #write.csv(properties , name, row.names = FALSE)
    
    rm(properties)
    properties = data.frame(mldr.teste$measures)
    properties = cbind(fold = f, properties)
    measures.teste = rbind(measures.teste, properties)
    #name = paste(folderSave , "/properties-test-", f, ".csv", sep="")
    #write.csv(properties , name, row.names = FALSE)
    
    rm(properties)
    properties = data.frame(mldr.val$measures)
    properties = cbind(fold = f, properties)
    measures.val = rbind(measures.val, properties)
    #name = paste(folderSave , "/properties-val-", f, ".csv", sep="")
    #write.csv(properties , name, row.names = FALSE)
    
    rm(properties)
    properties = data.frame(mldr.tv$measures)
    properties = cbind(fold = f, properties)
    measures.tv = rbind(measures.tv, properties)
    #name = paste(folderSave , "/properties-tv-", f, ".csv", sep="")
    #write.csv(properties , name, row.names = FALSE)
    
    
    ##########################################################################  
    # name = paste(folderSave , "/plot-train-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.treino))
    # dev.off()
    # cat("\n")
    # 
    # name = paste(folderSave , "/plot-test-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.teste))
    # dev.off()
    # cat("\n")
    # 
    # name = paste(folderSave , "/plot-val-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.val))
    # dev.off()
    # cat("\n")
    # 
    # name = paste(folderSave , "/plot-tv-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.tv))
    # dev.off()
    # cat("\n")
    
    
    f = f + 1
    gc()
  }
  
  name = paste(folder , "/properties-tv.csv", sep="")
  write.csv(measures.tv[-1,], name, row.names = FALSE)
  
  name = paste(folder , "/properties-test.csv", sep="")
  write.csv(measures.teste[-1,], name, row.names = FALSE)
  
  name = paste(folder , "/properties-train.csv", sep="")
  write.csv(measures.treino[-1,], name, row.names = FALSE)
  
  name = paste(folder , "/properties-val.csv", sep="")
  write.csv(measures.val[-1,], name, row.names = FALSE)
  
  
}


###############################################################################
#
###############################################################################
directories <- function(parameters){
  
  #library(here)
  #library(stringr)
  #FolderRoot <- here::here()
  #FolderScripts <- here::here("R")
  
  
  retorno = list()
  
  #############################################################################
  # RESULTS FOLDER:                                                           #
  # Parameter from command line. This folder will be delete at the end of the #
  # execution. Other folder is used to store definitely the results.          #
  # Example: "/dev/shm/result"; "/scratch/result"; "/tmp/result"              #
  #############################################################################
  if(dir.exists(parameters$Config.File$Folder.Results) == TRUE){
    setwd(folderResults)
    dir_folderResults = dir(folderResults)
    n_folderResults = length(dir_folderResults)
  } else {
    dir.create(folderResults)
    setwd(folderResults)
    dir_folderResults = dir(folderResults)
    n_folderResults = length(dir_folderResults)
  }
  retorno$FolderResults = parameters$Config.File$Folder.Results
  
  #############################################################################
  #
  #############################################################################
  FolderScripts = paste(FolderRoot, "/R", sep="")
  if(dir.exists(FolderScripts ) == TRUE){
    setwd(FolderScripts)
    dir_FolderScripts  = dir(FolderScripts )
    n_FolderScripts = length(dir_FolderScripts )
  } else {
    dir.create(FolderScripts)
    setwd(FolderScripts)
    dir_FolderScripts  = dir(FolderScripts )
    n_FolderScripts = length(dir_FolderScripts )
  }
  retorno$FolderScripts = FolderScripts
  
  
  #############################################################################
  #
  #############################################################################
  FolderReports = paste(FolderRoot, "/Reports", sep="")
  if(dir.exists(FolderReports  ) == TRUE){
    setwd(FolderReports )
    dir_FolderReports = dir(FolderReports)
    n_FolderReports = length(dir_FolderReports )
  } else {
    dir.create(FolderReports)
    setwd(FolderReports)
    dir_FolderReports = dir(FolderReports)
    n_FolderReports = length(dir_FolderReports )
  }
  retorno$FolderReports = FolderReports
  
  
  
  #############################################################################
  #
  #############################################################################
  folderUtils = paste(FolderRoot, "/Utils", sep="")
  if(dir.exists(folderUtils) == TRUE){
    setwd(folderUtils)
    dir_folderUtils = dir(folderUtils)
    n_folderUtils = length(dir_folderUtils)
  } else {
    dir.create(folderUtils)
    setwd(folderUtils)
    dir_folderUtils = dir(folderUtils)
    n_folderUtils = length(dir_folderUtils)
  }
  retorno$FolderUtils = folderUtils
  
  #############################################################################
  #
  #############################################################################
  folderPython = paste(FolderRoot, "/Python", sep="")
  if(dir.exists(folderPython) == TRUE){
    setwd(folderPython)
    dir_folderPython = dir(folderPython)
    n_folderPython = length(dir_folderPython)
  } else {
    dir.create(folderPython)
    setwd(folderPython)
    dir_folderPython = dir(folderPython)
    n_folderPython = length(dir_folderPython)
  }
  retorno$FolderPython = folderPython
  
  
  ###############################################################################
  #
  ###############################################################################
  folderGlobal = paste(folderResults, "/Global", sep="")
  if(dir.exists(folderGlobal) == TRUE){
    setwd(folderGlobal)
    dir_folderGlobal = dir(folderGlobal)
    n_folderGlobal = length(dir_folderGlobal)
  } else {
    dir.create(folderGlobal)
    setwd(folderGlobal)
    dir_folderGlobal = dir(folderGlobal)
    n_folderGlobal = length(dir_folderGlobal)
  }
  retorno$FolderGlobal = folderGlobal
  
  
  ###############################################################################
  #
  ###############################################################################
  folderDataset = paste(folderResults, "/Dataset", sep="")
  if(dir.exists(folderDataset) == TRUE){
    setwd(folderDataset)
    dir_folderDataset = dir(folderDataset)
    n_folderDataset = length(dir_folderDataset)
  } else {
    dir.create(folderDataset)
    setwd(folderDataset)
    dir_folderDataset = dir(folderDataset)
    n_folderDataset = length(dir_folderDataset)
  }
  retorno$FolderDataset = folderDataset
  
  
  ###############################################################################
  #
  ###############################################################################
  folderDatasetX = paste(folderDataset, "/", dataset_name, sep="")
  if(dir.exists(folderDatasetX) == TRUE){
    setwd(folderDatasetX)
    dir_folderDatasetX = dir(folderDatasetX)
    n_folderDatasetX = length(dir_folderDatasetX)
  } else {
    dir.create(folderDatasetX)
    setwd(folderDatasetX)
    dir_folderDatasetX = dir(folderDatasetX)
    n_folderDatasetX = length(dir_folderDatasetX)
  }
  retorno$FolderDatasetX = folderDatasetX
  
  
  ###############################################################################
  #
  ###############################################################################
  folderCV = paste(folderDatasetX, "/CrossValidation", sep="")
  if(dir.exists(folderCV) == TRUE){
    setwd(folderCV)
    dir_folderCV = dir(folderCV)
    n_folderCV = length(dir_folderCV)
  } else {
    dir.create(folderCV)
    setwd(folderCV)
    dir_folderCV = dir(folderCV)
    n_folderCV = length(dir_folderCV)
  }
  retorno$FolderCV = folderCV
  
  
  ###############################################################################
  #
  ###############################################################################
  folderCVTR = paste(folderCV, "/Tr", sep="")
  if(dir.exists(folderCVTR) == TRUE){
    setwd(folderCVTR)
    dir_folderCVTR = dir(folderCVTR)
    n_folderCVTR = length(dir_folderCVTR)
  } else {
    dir.create(folderCVTR)
    setwd(folderCVTR)
    dir_folderCVTR = dir(folderCVTR)
    n_folderCVTR = length(dir_folderCVTR)
  }
  retorno$FolderCVTR = folderCVTR
  
  
  ###############################################################################
  #
  ###############################################################################
  folderCVTS = paste(folderCV, "/Ts", sep="")
  if(dir.exists(folderCVTS) == TRUE){
    setwd(folderCVTS)
    dir_folderCVTS = dir(folderCVTS)
    n_folderCVTS = length(dir_folderCVTS)
  } else {
    dir.create(folderCVTS)
    setwd(folderCVTS)
    dir_folderCVTS = dir(folderCVTS)
    n_folderCVTS = length(dir_folderCVTS)
  }
  retorno$FolderCVTS = folderCVTS
  
  
  ###############################################################################
  #
  ###############################################################################
  folderCVVL = paste(folderCV, "/Vl", sep="")
  if(dir.exists(folderCVVL) == TRUE){
    setwd(folderCVVL)
    dir_folderCVVL = dir(folderCVVL)
    n_folderCVVL = length(dir_folderCVVL)
  } else {
    dir.create(folderCVVL)
    setwd(folderCVVL)
    dir_folderCVVL = dir(folderCVVL)
    n_folderCVVL = length(dir_folderCVVL)
  }
  retorno$FolderCVVL = folderCVVL
  
  ###############################################################################
  #
  ###############################################################################
  folderLabelSpace = paste(folderDatasetX, "/LabelSpace", sep="")
  if(dir.exists(folderLabelSpace) == TRUE){
    setwd(folderLabelSpace)
    dir_folderLabelSpace = dir(folderLabelSpace)
    n_folderLabelSpace = length(dir_folderLabelSpace)
  } else {
    dir.create(folderLabelSpace)
    setwd(folderLabelSpace)
    dir_folderLabelSpace = dir(folderLabelSpace)
    n_folderLabelSpace = length(dir_folderLabelSpace)
  }
  retorno$FolderLabelSpace = folderLabelSpace
  
  
  ###############################################################################
  #
  ###############################################################################
  folderNamesLabels = paste(folderDatasetX, "/NamesLabels", sep="")
  if(dir.exists(folderNamesLabels) == TRUE){
    setwd(folderNamesLabels)
    dir_folderNamesLabels = dir(folderNamesLabels)
    n_folderNamesLabels = length(dir_folderNamesLabels)
  } else {
    dir.create(folderNamesLabels)
    setwd(folderNamesLabels)
    dir_folderNamesLabels = dir(folderNamesLabels)
    n_folderNamesLabels = length(dir_folderNamesLabels)
  }
  retorno$FolderNamesLabels = folderNamesLabels
  
  
  ############################################################################
  return(retorno)
  gc()
  
}



##############################################################################
# 
##############################################################################
infoDataSet <- function(dataset){
  retorno = list()
  retorno$id = dataset$ID
  retorno$name = dataset$Name
  retorno$instances = dataset$Instances
  retorno$inputs = dataset$Inputs
  retorno$labels = dataset$Labels
  retorno$LabelsSets = dataset$LabelsSets
  retorno$single = dataset$Single
  retorno$maxfreq = dataset$MaxFreq
  retorno$card = dataset$Card
  retorno$dens = dataset$Dens
  retorno$mean = dataset$Mean
  retorno$scumble = dataset$Scumble
  retorno$tcs = dataset$TCS
  retorno$attStart = dataset$AttStart
  retorno$attEnd = dataset$AttEnd
  retorno$labStart = dataset$LabelStart
  retorno$labEnd = dataset$LabelEnd
  return(retorno)
  gc()
}


##############################################################################
# 
##############################################################################
roc.curva <- function(f, y_pred, test, Folder, nome){
  
  #####################################################################
  y_pred= sapply(y_pred, function(x) as.numeric(as.character(x)))
  res = mldr_evaluate(test, y_pred)
  
  ###############################################################
  # PLOTANDO ROC CURVE
  # name = paste(Folder, "/", nome, "-roc.pdf", sep="")
  # pdf(name, width = 10, height = 8)
  # print(plot(res$roc, print.thres = 'best', print.auc=TRUE, 
  #            print.thres.cex=0.7, grid = TRUE, identity=TRUE,
  #            axes = TRUE, legacy.axes = TRUE, 
  #            identity.col = "#a91e0e", col = "#1161d5",
  #            main = paste("fold ", f, " ", nome, sep="")))
  # dev.off()
  # cat("\n")
  
  ###############################################################
  write.csv(as.numeric(res$roc$auc), paste(Folder, "/", nome, "-roc-auc.csv", sep=""))
  write.csv(as.numeric(res$macro_auc), paste(Folder, "/", nome, "-roc-auc-macro.csv", sep=""))
  write.csv(as.numeric(res$micro_auc), paste(Folder, "/", nome, "-roc-auc-micro.csv", sep=""))
  
  
  ###############################################################
  # SALVANDO AS INFORMAÇÕES DO ROC SEPARADAMENTE
  name = paste(Folder, "/", nome, "-roc-1.txt", sep="")
  output.file <- file(name, "wb")
  
  write(" ", file = output.file, append = TRUE)
  write("percent: ", file = output.file, append = TRUE)
  write(res$roc$percent, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("sensitivities: ", file = output.file, append = TRUE)
  write(res$roc$sensitivities, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("specificities: ", file = output.file, append = TRUE)
  write(res$roc$specificities, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("thresholds: ", file = output.file, append = TRUE)
  write(res$roc$thresholds, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("direction: ", file = output.file, append = TRUE)
  write(res$roc$direction, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("cases: ", file = output.file, append = TRUE)
  write(res$roc$cases, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("controls: ", file = output.file, append = TRUE)
  write(res$roc$controls, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("auc: ", file = output.file, append = TRUE)
  write(res$roc$auc, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("original predictor: ", file = output.file, append = TRUE)
  write(res$roc$original.predictor, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("original response: ", file = output.file, append = TRUE)
  write(res$roc$original.response, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("predictor: ", file = output.file, append = TRUE)
  write(res$roc$predictor, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("response: ", file = output.file, append = TRUE)
  write(res$roc$response, file = output.file, append = TRUE)
  
  write(" ", file = output.file, append = TRUE)
  write("levels: ", file = output.file, append = TRUE)
  write(res$roc$levels, file = output.file, append = TRUE)
  
  close(output.file)
  
  ###############################################################
  # SALVANDO AS OUTRAS INFORMAÇÕES
  name = paste(Folder, "/", nome, "-roc-2.txt", sep="")
  sink(name, type = "output")
  print(res$roc)
  cat("\n\n")
  str(res)
  sink()
}



##############################################################################
# 
##############################################################################
matrix.confusao <- function(true, pred, type, salva, nomes.rotulos){ 
  
  bipartition = data.frame(true, pred)
  
  num.instancias = nrow(bipartition)
  num.rotulos = ncol(true) # número de rótulos do conjunto
  
  num.positive.instances = apply(bipartition, 2, sum) # número de instâncias positivas
  num.negative.instances = num.instancias - num.positive.instances   # número de instâncias negativas  # salvando
  
  res = rbind(num.positive.instances, num.negative.instances)
  name = paste(salva, "/", type, "-ins-pn.csv", sep="")
  write.csv(res, name)
  
  true_1 = data.frame(ifelse(true==1,1,0)) # calcular rótulo verdadeiro igual a 1
  total_true_1 = apply(true_1, 2, sum)
  
  true_0 = data.frame(ifelse(true==0,1,0)) # calcular rótulo verdadeiro igual a 0
  total_true_0 = apply(true_0, 2, sum)
  
  pred_1 = data.frame(ifelse(pred==1,1,0)) # calcular rótulo predito igual a 1
  total_pred_1 = apply(pred_1, 2, sum)
  
  pred_0 = data.frame(ifelse(pred==0,1,0)) # calcular rótulo verdadeiro igual a 0
  total_pred_0 = apply(pred_0, 2, sum)
  
  matriz_totais = cbind(total_true_0, total_true_1, total_pred_0, total_pred_1)
  row.names(matriz_totais) = nomes.rotulos
  name = paste(salva, "/", type, "-trues-preds.csv", sep="")
  write.csv(matriz_totais, name)
  
  # Verdadeiro Positivo: O modelo previu 1 e a resposta correta é 1
  TPi  = data.frame(ifelse((true_1 & true_1),1,0))
  tpi = paste(nomes.rotulos, "-TP", sep="")
  names(TPi) = tpi
  
  # Verdadeiro Negativo: O modelo previu 0 e a resposta correta é 0
  TNi  = data.frame(ifelse((true_0 & pred_0),1,0))
  tni = paste(nomes.rotulos, "-TN", sep="")
  names(TNi) = tni
  
  # Falso Positivo: O modelo previu 1 e a resposta correta é 0
  FPi  = data.frame(ifelse((true_0 & pred_1),1,0))
  fpi = paste(nomes.rotulos, "-FP", sep="")
  names(FPi) = fpi
  
  # Falso Negativo: O modelo previu 0 e a resposta correta é 1
  FNi  = data.frame(ifelse((true_1 & pred_0),1,0))
  fni = paste(nomes.rotulos, "-FN", sep="")
  names(FNi) = fni
  
  fpnt = data.frame(TPi, FPi, FNi, TNi)
  name = paste(salva, "/", type, "-tfpn.csv", sep="")
  write.csv(fpnt, name, row.names = FALSE)
  
  # total de verdadeiros positivos
  TPl = apply(TPi, 2, sum)
  tpl = paste(nomes.rotulos, "-TP", sep="")
  names(TPl) = tpl
  
  # total de verdadeiros negativos
  TNl = apply(TNi, 2, sum)
  tnl = paste(nomes.rotulos, "-TN", sep="")
  names(TNl) = tnl
  
  # total de falsos negativos
  FNl = apply(FNi, 2, sum)
  fnl = paste(nomes.rotulos, "-FN", sep="")
  names(FNl) = fnl
  
  # total de falsos positivos
  FPl = apply(FPi, 2, sum)
  fpl = paste(nomes.rotulos, "-FP", sep="")
  names(FPl) = fpl
  
  matriz_confusao_por_rotulos = data.frame(TPl, FPl, FNl, TNl)
  colnames(matriz_confusao_por_rotulos) = c("TP","FP", "FN", "TN")
  row.names(matriz_confusao_por_rotulos) = nomes.rotulos
  name = paste(salva, "/", type, "-matrix-confusion.csv", sep="")
  write.csv(matriz_confusao_por_rotulos, name)
}

##############################################################
#
##############################################################
avaliacao <- function(f, y_true, y_pred, salva, nome){
  
  #salva.0 = paste(salva, "/", nome, "-conf-mat.txt", sep="")
  #sink(file=salva.0, type="output")
  confmat = multilabel_confusion_matrix(y_true, y_pred)
  #print(confmat)
  #sink()
  
  resConfMat = multilabel_evaluate(confmat)
  resConfMat = data.frame(resConfMat)
  names(resConfMat) = paste("Fold-", f, sep="")
  salva.1 = paste(salva, "/", nome, ".csv", sep="")
  write.csv(resConfMat, salva.1)
  
  conf.mat = data.frame(confmat$TPl, confmat$FPl,
                        confmat$FNl, confmat$TNl)
  names(conf.mat) = c("TP", "FP", "FN", "TN")
  conf.mat.perc = data.frame(conf.mat/nrow(y_true$dataset))
  names(conf.mat.perc) = c("TP.perc", "FP.perc", "FN.perc", "TN.perc")
  wrong = conf.mat$FP + conf.mat$FN
  wrong.perc = wrong/nrow(y_true$dataset)
  correct = conf.mat$TP + conf.mat$TN
  correct.perc = correct/nrow(y_true$dataset)
  conf.mat.2 = data.frame(conf.mat, conf.mat.perc, wrong, correct, 
                          wrong.perc, correct.perc)
  salva.2 = paste(salva, "/", nome, "-utiml.csv", sep="")
  #write.csv(conf.mat.2, salva.2)
  
  
}




###############################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                #
# Thank you very much!                                                        #
###############################################################################
