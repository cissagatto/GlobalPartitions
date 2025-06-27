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
gather.files.python <- function(parameters){
  
  # f = 1
  foldsParalel <- foreach(f = 1:parameters$Config.File$Number.Folds) %dopar% {
    # while(f<=number_folds){
    
    cat("\nFold: ", f)
    
    ###########################################################################
    # library(here)
    # library(stringr)
    # FolderRoot <- here::here()
    # FolderScripts <- here::here("R")
    
    
    
    ###########################################################################
    FolderSplit = paste(parameters$Directories$FolderGlobal,
                        "/Split-", f, sep="")
    if(dir.exists(FolderSplit)==FALSE){dir.create(FolderSplit)}
    
    ###########################################################################
    # names files
    nome.tr.csv = paste(parameters$Config.File$Dataset.Name , 
                        "-Split-Tr-", f, ".csv", sep="")
    nome.ts.csv = paste(parameters$Config.File$Dataset.Name, 
                        "-Split-Ts-", f, ".csv", sep="")
    nome.vl.csv = paste(parameters$Config.File$Dataset.Name, 
                        "-Split-Vl-", f, ".csv", sep="")

    ###########################################################################    
    # train
    setwd(parameters$Directories$FolderCVTR)
    if(file.exists(nome.tr.csv) == TRUE){
      setwd(parameters$Directories$FolderCVTR)
      copia = paste(parameters$Directories$FolderCVTR, "/", 
                    nome.tr.csv, sep="")
      cola = paste(FolderSplit, "/", nome.tr.csv, sep="")
      file.copy(copia, cola, overwrite = TRUE)
    }
    
    ###########################################################################
    # test
    setwd(parameters$Directories$FolderCVTS)
    if(file.exists(nome.ts.csv) == TRUE){
      setwd(parameters$Directories$FolderCVTS)
      copia = paste(parameters$Directories$FolderCVTS, "/", 
                    nome.ts.csv, sep="")
      cola = paste(FolderSplit, "/", nome.ts.csv, sep="")
      file.copy(copia, cola, overwrite = TRUE)
    }
    
    ###########################################################################
    # validation
    setwd(parameters$Directories$FolderCVVL)
    if(file.exists(nome.vl.csv) == TRUE){
      setwd(parameters$Directories$FolderCVVL)
      copia = paste(parameters$Directories$FolderCVVL, "/", 
                    nome.vl.csv, sep="")
      cola = paste(FolderSplit, "/", nome.vl.csv, sep="")
      file.copy(copia, cola, overwrite = TRUE)
    }
    
    # f = f + 1
    gc()
  }
  
  gc()
  cat("\n###############################################################")
  cat("\n# GLOBAL RF: END OF THE GATHER FILES FOLDS FUNCTION           #")
  cat("\n###############################################################")
  cat("\n\n")
}


##############################################################################
#
##############################################################################
execute.global.python <- function(parameters){
  
  f = 2

  RfGlobalParalel <- foreach(f = 1:parameters$Config.File$Number.Folds) %dopar%{
  # while(f<=parameters$Config.File$Number.Folds){
    
    #########################################################################
    cat("\nFold: ", f)
    
    ##########################################################################
    # library(here)
    # library(stringr)
    # FolderRoot <- here::here()
    # FolderScripts <- here::here("R")
    # source(file.path(FolderScripts, "libraries.R"))
    # source(file.path(FolderScripts, "utils.R"))
    
    source(file.path(parameters$Directories$FolderScripts, "libraries.R"))
    source(file.path(parameters$Directories$FolderScripts, "utils.R"))
    
    
    ###########################################################################
    # /tmp/gr-emotions/Global/Split-1
    FolderSplit = paste(parameters$Directories$FolderGlobal, 
                        "/Split-", f, sep="")
    if(dir.exists(FolderSplit)==FALSE){dir.create(FolderSplit)}
    
    
    ###########################################################################
    # names files
    nome.tr.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name , 
                        "-Split-Tr-", f, ".csv", sep="")
    nome.ts.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name, 
                        "-Split-Ts-", f, ".csv", sep="")
    nome.vl.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name, 
                        "-Split-Vl-", f, ".csv", sep="")
    
    
    ##########################################################################
    train = data.frame(read.csv(nome.tr.csv))
    test = data.frame(read.csv(nome.ts.csv))
    val = data.frame(read.csv(nome.vl.csv))
    tv = rbind(train, val)
    
    
    ##########################################################################
    labels.indices = seq(parameters$Dataset.Info$LabelStart, 
                         parameters$Dataset.Info$LabelEnd, by=1)
    
    
    ##########################################################################
    mldr.treino = mldr_from_dataframe(train, labelIndices = labels.indices)
    mldr.test = mldr_from_dataframe(test, labelIndices = labels.indices)
    mldr.tv = mldr_from_dataframe(tv, labelIndices = labels.indices)
    
    
    ##################################################################
    str.execute = paste("python3 ", parameters$Directories$FolderPython,
                        "/global.py ", 
                        nome.tr.csv, " ",
                        nome.vl.csv,  " ",
                        nome.ts.csv, " ", 
                        start = as.numeric(parameters$Dataset.Info$AttEnd), " ",
                        FolderSplit, " ", 
                        fold = f,
                        sep="")
    
    # /tmp/gr-emotions/Dataset/emotions/CrossValidation/Tr
    
    # EXECUTA
    start <- proc.time()
    res = print(system(str.execute))
    tempo = data.matrix((proc.time() - start))
    tempo = data.frame(t(tempo))
    write.csv(tempo, paste(FolderSplit, "/runtime-fold.csv", sep=""))
    
    if(res!=0){
      break
    }
    
    # f = f + 1
    gc()
  }
  
  gc()
  cat("\n############################################################")
  cat("\n# GLOBAL RF: END OF FUNCTION EXECUTE                       #")
  cat("\n############################################################")
  cat("\n\n")
}


############################################################################
#
############################################################################
evaluate.global.python <- function(parameters){
  
  # f = 1
  avaliaParalel <- foreach (f = 1:parameters$Config.File$Number.Folds) %dopar%{
  # while(f<=parameters$Config.File$Number.Folds){
    
    #########################################################################
    cat("\nFold: ", f)
    
    ##########################################################################
    # library(here)
    # library(stringr)
    # FolderRoot <- here::here()
    # FolderScripts <- here::here("R")
    # source(file.path(FolderScripts, "libraries.R"))
    # source(file.path(FolderScripts, "utils.R"))
    
    ###########################################################################
    source(file.path(parameters$Directories$FolderScripts, "libraries.R"))
    source(file.path(parameters$Directories$FolderScripts, "utils.R"))
    
    ##########################################################################
    train.file.name = paste(parameters$Directories$FolderCVTR, "/", 
                            parameters$Config$Dataset.Name, 
                            "-Split-Tr-", f , ".csv", sep="")
    
    test.file.name = paste(parameters$Directories$FolderCVTS, "/", 
                           parameters$Config$Dataset.Name, 
                           "-Split-Ts-", f, ".csv", sep="")
    
    val.file.name = paste(parameters$Directories$FolderCVVL, "/", 
                          parameters$Config$Dataset.Name, 
                          "-Split-Vl-", f , ".csv", sep="")
    
    ##########################################################################
    train = data.frame(read.csv(train.file.name))
    test = data.frame(read.csv(test.file.name))
    val = data.frame(read.csv(val.file.name))
    tv = rbind(train, val)
    
    
    ##########################################################################
    labels.indices = seq(parameters$Dataset.Info$LabelStart, 
                         parameters$Dataset.Info$LabelEnd, by=1)
    
    ##########################################################################
    mldr.treino = mldr_from_dataframe(train, labelIndices = labels.indices)
    mldr.teste = mldr_from_dataframe(test, labelIndices = labels.indices)
    mldr.val = mldr_from_dataframe(val, labelIndices = labels.indices)
    mldr.tv = mldr_from_dataframe(tv, labelIndices = labels.indices)
    
    
    ###########################################################################
    FolderSplit = paste(parameters$Directories$FolderGlobal, "/Split-", f, sep="")
    if(dir.exists(FolderSplit)==FALSE){dir.create(FolderSplit)}
    
    
    #####################################################################
    nome.true = paste(FolderSplit, "/y_true.csv", sep="")
    nome.pred.proba = paste(FolderSplit, "/y_pred_proba.csv", sep="")
    nome.pred.bin = paste(FolderSplit, "/y_pred_bin.csv", sep="")
    
    
    #####################################################################
    y_true = data.frame(read.csv(nome.true))
    y_pred_proba = data.frame(read.csv(nome.pred.proba))
    y_pred_bin = data.frame(read.csv(nome.pred.bin))
    
    
    ##########################################################################
    y.true.2 = data.frame(sapply(y_true, function(x) as.numeric(as.character(x))))
    y.true.3 = mldr_from_dataframe(y.true.2, 
                                   labelIndices = seq(1,ncol(y.true.2)), 
                                   name = "y.true.2")
    y_pred_bin = sapply(y_pred_bin, function(x) as.numeric(as.character(x)))
    y_pred_proba = sapply(y_pred_proba, function(x) as.numeric(as.character(x)))
    
    
    ########################################################################
    y_threshold_05 <- data.frame(as.matrix(fixed_threshold(y_pred_proba,
                                                           threshold = 0.5)))
    write.csv(y_threshold_05, 
              paste(FolderSplit, "/y_pred_thr05.csv", sep=""),
              row.names = FALSE)
    
    ########################################################################
    y_threshold_card = lcard_threshold(as.matrix(y_pred_proba), 
                                       mldr.tv$measures$cardinality,
                                       probability = F)
    write.csv(y_threshold_card, 
              paste(FolderSplit, "/y_pred_thrLC.csv", sep=""),
              row.names = FALSE)
    
    
    ##########################################################################    
    avaliacao(f = f, y_true = y.true.3, y_pred = y_pred_proba,
              salva = FolderSplit, nome = "results-utiml")
    
    # avaliacao(f = f, y_true = y.true.3, y_pred = y_pred_bin,
    #           salva = FolderSplit, nome = "pred-bin")
    # 
    # avaliacao(f = f, y_true = y.true.3, y_pred = y_threshold_05,
    #           salva = FolderSplit, nome = "thr-05")
    # 
    # avaliacao(f = f, y_true = y.true.3, y_pred = y_threshold_card,
    #           salva = FolderSplit, nome = "thr-lc")
    
    ###########################################################################
    # names files
    nome.tr.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name , 
                        "-Split-Tr-", f, ".csv", sep="")
    nome.ts.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name, 
                        "-Split-Ts-", f, ".csv", sep="")
    nome.vl.csv = paste(FolderSplit, "/", 
                        parameters$Config.File$Dataset.Name, 
                        "-Split-Vl-", f, ".csv", sep="")
    
    system(paste0("rm -r ", nome.tr.csv))
    system(paste0("rm -r ", nome.ts.csv))
    system(paste0("rm -r ", nome.vl.csv))
    
    # f = f + 1
    gc()
  }
  
  gc()
  cat("\n##################################")
  cat("\n# END FUNCTION EVALUATE          #")
  cat("\n##################################")
  cat("\n\n\n\n")
}



###########################################################################
#
###########################################################################
gather.eval.python.silho <- function(parameters){
  
  final.runtime.r = data.frame()
  final.runtime.p = data.frame()
  final.results = data.frame(apagar=c(0))
  total.model.size = data.frame()
  
  f = 1
  while(f<=parameters$Config$Number.Folds){
    
    cat("\nFold: ", f)
    
    #########################################################################
    folderSplit = paste(parameters$Directories$FolderGlobal,
                        "/Split-", f, sep="")
    
    #########################################################################
    res.python = data.frame(read.csv(paste(folderSplit, 
                                           "/results-python.csv", sep="")))
    names(res.python) = c("Measures", paste0("Fold",f))
    
    #########################################################################
    res.utiml = data.frame(read.csv(paste(folderSplit, 
                                          "/results-utiml.csv", sep="")))
    names(res.utiml) = c("Measures", paste0("Fold",f))
    
    #########################################################################
    resultados = rbind(res.python, res.utiml)
    final.results = cbind(final.results, resultados)
    
    #########################################################################
    res.model.size = data.frame(read.csv(paste(folderSplit, 
                                               "/model-sizes.csv", sep="")))
    names(res.model.size) = c("Format", "Bytes")
    resultado = data.frame(fold = paste0("fold",f), res.model.size)
    total.model.size = rbind(total.model.size, resultado)
    
    #########################################################################
    res.runtime.fold = data.frame(read.csv(paste(folderSplit, 
                                                 "/runtime-fold.csv", sep="")))
    res.runtime.fold = res.runtime.fold[,-1]
    res.runtime.fold = data.frame(fold=f, res.runtime.fold)
    final.runtime.r = rbind(final.runtime.r, res.runtime.fold)
    
    #########################################################################
    res.runtime.python = data.frame(read.csv(paste(folderSplit, 
                                                   "/runtime-python.csv", sep="")))
    names(res.runtime.python) = c("Process", "Time")
    res.runtime.python = data.frame(fold = paste0("fold",f), res.runtime.python)
    final.runtime.p = rbind(final.runtime.p, res.runtime.python)
    
    #################################
    # /tmp/gr-emotions/Global/Split-1
    print(system(paste0("rm -r ", folderSplit, "/results-python.csv", sep="")))
    print(system(paste0("rm -r ", folderSplit, "/results-utiml.csv", sep="")))
    print(system(paste0("rm -r ", folderSplit, "/model-sizes.csv", sep="")))
    print(system(paste0("rm -r ", folderSplit, "/runtime-python.csv", sep="")))
    print(system(paste0("rm -r ", folderSplit, "/runtime-fold.csv", sep="")))
    
    f = f + 1
    gc()
  } 
  
  
  setwd(parameters$Directories$FolderGlobal)
  final.results <- final.results[, !duplicated(colnames(final.results))]
  final.results = final.results[,-1]
  write.csv(final.results, "performance.csv")
  
  write.csv(total.model.size, "model-size.csv", row.names = FALSE)
  write.csv(final.runtime.r, "runtime-r.csv", row.names = FALSE)
  write.csv(final.runtime.p, "runtime-p.csv", row.names = FALSE)
  
  gc()
  cat("\n########################################################")
  cat("\n# END EVALUATED                                        #") 
  cat("\n########################################################")
  cat("\n\n\n\n")
}

###############################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                #
# Thank you very much!                                                        #
###############################################################################
