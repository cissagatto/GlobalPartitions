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
#' Convert CSV files to ARFF format using a Java converter
#'
#' @description
#' This function calls a Java JAR file (`R_csv_2_arff.jar`) to convert a CSV dataset
#' into an ARFF file format (used by Weka and other machine learning tools).
#' It builds the system command dynamically and executes it from within R.
#'
#' @details
#' The function assumes that the Java JAR converter (`R_csv_2_arff.jar`) is located
#' in the folder specified by `FolderUtils`. The user must have Java properly installed
#' and accessible through the system PATH.
#'
#' @param arg1 Character. The path to the input CSV file to be converted.
#' @param arg2 Character. The path or name of the output ARFF file to be generated.
#' @param arg3 Character. Additional parameters to be passed to the Java converter.
#' @param FolderUtils Character. The directory containing the `R_csv_2_arff.jar` file.
#'
#' @return
#' The function prints the result of the system command execution to the console.
#' It does not return any R object (invisible return of `NULL`).
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' FolderUtils <- "/home/user/utils"
#' input_csv <- "/home/user/data/sample.csv"
#' output_arff <- "/home/user/data/sample.arff"
#'
#' converteArff(input_csv, output_arff, "", FolderUtils)
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [system()] for executing system commands in R,
#' [paste()] for string concatenation.
#'
#' @note
#' Make sure Java is installed and available in your system environment.
#' The JAR file `R_csv_2_arff.jar` must exist in the specified `FolderUtils` directory.
#'
#' @export
converteArff <- function(arg1, arg2, arg3, FolderUtils){  
  str = paste("java -jar ", FolderUtils,  "/R_csv_2_arff.jar ", 
              arg1, " ", arg2, " ", arg3, sep="")
  print(system(str))
  cat("\n\n")  
}




###############################################################################
#' Compute and export multilabel dataset properties for each fold
#'
#' @description
#' This function calculates and exports detailed multilabel dataset properties,
#' statistics, and distributions for each cross-validation fold (train, test, validation, and train+validation).
#' It relies on the `mldr` package to extract multilabel measures and label information,
#' saving multiple summary and diagnostic CSV files for analysis.
#'
#' @details
#' The function processes each fold defined in `parameters$Config.File$Number.Folds` by:
#' - Reading split files for training, testing, validation, and combined sets (train + validation).
#' - Computing label-based statistics such as mean, standard deviation, quantiles, and positive/negative instance counts.
#' - Extracting multilabel measures (e.g., label cardinality, density, imbalance ratio, etc.) using the `mldr` package.
#' - Saving multiple CSV summaries per fold (summary statistics, label frequencies, labelsets, etc.).
#' - Storing global property summaries across folds for all dataset partitions.
#'
#' Additionally, the function identifies labels with zero frequency across folds and reports them.
#'
#' @param parameters A list object containing all configuration and directory information required for processing.
#' The expected structure includes:
#' \itemize{
#'   \item \code{parameters$Config.File$Number.Folds} – Number of cross-validation folds.
#'   \item \code{parameters$Config.File$Dataset.Name} – Name of the dataset being processed.
#'   \item \code{parameters$Directories$FolderResults} – Output directory for saving results.
#'   \item \code{parameters$Directories$FolderCVTR}, \code{FolderCVTS}, \code{FolderCVVL} – Paths to training, testing, and validation split CSVs.
#'   \item \code{parameters$Directories$FolderNamesLabels} – Path containing the dataset’s label names CSV file.
#'   \item \code{parameters$Dataset.Info$LabelStart}, \code{parameters$Dataset.Info$LabelEnd} – Index range for label columns.
#' }
#'
#' @return
#' A data frame containing information about labels with zero frequency across folds.
#' The function also writes multiple CSV files to disk:
#' \itemize{
#'   \item Per-fold summaries (e.g., `summary-train-<fold>.csv`, `summary-test-<fold>.csv`).
#'   \item Label frequencies (`instances-pos-neg-<fold>.csv`, `labels-max-min-<fold>.csv`).
#'   \item Labelsets distributions (`labelsets-train-<fold>.csv`, etc.).
#'   \item Aggregated property summaries (`properties-train.csv`, `properties-test.csv`, etc.).
#' }
#'
#' @examples
#' \dontrun{
#' # Example usage
#' library(mldr)
#' 
#' parameters <- list(
#'   Config.File = list(
#'     Number.Folds = 10,
#'     Dataset.Name = "emotions"
#'   ),
#'   Directories = list(
#'     FolderResults = "/home/user/results",
#'     FolderCVTR = "/home/user/splits/train",
#'     FolderCVTS = "/home/user/splits/test",
#'     FolderCVVL = "/home/user/splits/val",
#'     FolderNamesLabels = "/home/user/data"
#'   ),
#'   Dataset.Info = list(
#'     LabelStart = 7,
#'     LabelEnd = 12
#'   )
#' )
#'
#' properties.datasets(parameters)
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [mldr::mldr_from_dataframe()] for generating multilabel dataset objects,  
#' [apply()] and [write.csv()] for computing and saving statistics.
#'
#' @note
#' - The function assumes that all necessary split CSVs and label files exist in the provided directories.
#' - The `mldr` package must be installed and loaded before running this function.
#' - Folder paths are created automatically if missing.
#'
#' @import mldr
#' @importFrom dplyr arrange desc
#'
#' @export
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
  
  zeros = data.frame()
  
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
  
  folderProperties = paste(parameters$Directories$FolderResults, 
                 "", parameters$Dataset.Name,
                 "/Properties", sep="")
  if(dir.exists(folderProperties)==FALSE){dir.create(folderProperties)}
  
  
  f = 1
  while(f<=parameters$Config.File$Number.Folds){
    
    cat("\n\n\n%------------Fold [", f, "]------------%\n\n\n")
    
    ####################################################################
    folderSave = paste(folderProperties , "/Split-", f, sep="")
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
    
    
    ####################################################################
    nome = paste(parameters$Directories$FolderCVTS, 
                 "/", parameters$Config.File$Dataset.Name,
                 "-Split-Ts-", f, ".csv", sep="")
    teste = data.frame(read.csv(nome))
    
    
    ####################################################################
    nome = paste(parameters$Directories$FolderCVVL, 
                 "/", parameters$Config.File$Dataset.Name,
                 "-Split-Vl-", f, ".csv", sep="")
    val = data.frame(read.csv(nome))
    
    
    ####################################################################
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
    
    
    ####################################################
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
    labels.train = data.frame(mldr.treino$labels)
    name = paste(folderSave, "/labels-train-", f, ".csv", sep="")
    write.csv(labels.train, name)
    
    # if(any(labels.train$count == 0)) {
    #   zero_counts <- labels.train[labels.train$count == 0, ]
    #   zeros = rbind(zeros, zero_counts)
    #   cat("\n\ntem zeros\n\n")
    # } else {
    #   cat("\n\nnão tem zeros\n\n")
    # }
    
    labels.test = data.frame(mldr.teste$labels)
    name = paste(folderSave, "/labels-test-", f, ".csv", sep="")
    write.csv(labels.test, name)
    
    # if(any(labels.test$count == 0)) {
    #   zero_counts <- labels.test[labels.test$count == 0, ]
    #   zeros = rbind(zeros, zero_counts)
    #   cat("\n\ntem zeros\n\n")
    # } else {
    #   cat("\n\nnão tem zeros\n\n")
    # }
    
    
    labels.val = data.frame(mldr.val$labels)
    name = paste(folderSave, "/labels-val-", f, ".csv", sep="")
    write.csv(labels.val, name)
     
    # if(any(labels.val$count == 0)) {
    #   zero_counts <- labels.val[labels.val$count == 0, ]
    #   zeros = rbind(zeros, zero_counts)
    #   cat("\n\ntem zeros\n\n")
    # } else {
    #   cat("\n\nnão tem zeros\n\n")
    # }
    
    
    labels.tv = data.frame(mldr.tv$labels)
    name = paste(folderSave, "/labels-tv-", f, ".csv", sep="")
    write.csv(labels.tv, name)
    
    # if(any(labels.tv$count == 0)) {
    #   zero_counts <- labels.tv[labels.tv$count == 0, ]
    #   zeros = rbind(zeros, zero_counts)
    #   cat("\n\nTEM ZEROS\n\n")
    # } else {
    #   cat("\n\nnão tem zeros\n\n")
    # }
    
    
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
      
    # name = paste(folderSave , "/plot-test-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.teste))
    # dev.off()
    # cat("\n")
     
    # name = paste(folderSave , "/plot-val-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.val))
    # dev.off()
    # cat("\n")
     
    # name = paste(folderSave , "/plot-tv-fold-", f, ".pdf", sep="")
    # pdf(name, width = 10, height = 8)
    # print(plot(mldr.tv))
    # dev.off()
    # cat("\n")
    
    
    f = f + 1
    gc()
  }
  
  #zeros
  
  name = paste0(parameters$Directories$FolderResults, 
                "/Properties/properties-tv.csv")
  write.csv(data.frame(measures.tv[-1,]), name, row.names = FALSE)
  
  name = paste0(parameters$Directories$FolderResults , 
                "/Properties/properties-test.csv", sep="")
  write.csv(measures.teste[-1,], name, row.names = FALSE)
  
  name = paste0(parameters$Directories$FolderResults, 
                "/Properties/properties-train.csv", sep="")
  write.csv(measures.treino[-1,], name, row.names = FALSE)
  
  name = paste0(parameters$Directories$FolderResults , 
                "/Properties/properties-val.csv", sep="")
  write.csv(measures.val[-1,], name, row.names = FALSE)
  
  
}


###############################################################################
#' Create and manage the main directory structure for an experiment
#'
#' @description
#' This function sets up and verifies all necessary directories for an experiment,
#' based on configuration parameters provided in the `parameters` list.
#' It creates folders for results, reports, scripts, datasets, cross-validation splits,
#' label space, and other utilities.  
#' The function ensures that all folders exist and returns their paths in a structured list.
#'
#' @details
#' The folder structure is designed to support experiments involving data preprocessing,
#' model training, validation, and result storage.  
#' It includes subfolders such as:
#' - **Global**: global results
#' - **Dataset**: dataset
#' - **CrossValidation (Tr, Ts, Vl)**: training, testing, and validation folds
#' - **LabelSpace** and **NamesLabels**: label metadata and mappings
#'
#' The function automatically creates any folder that does not exist, ensuring that
#' subsequent processes have the required directory organization.
#'
#' @param parameters A list containing configuration information, including:
#'   - `Config.File$Folder.Results`: Path where results will be stored.
#'   - `dataset_name`: Name of the dataset used.
#'   - Other paths and variables defined in the experiment setup.
#'
#' @return
#' A list containing the paths of all created or verified folders:
#' \itemize{
#'   \item `FolderResults` – Main folder for storing experiment results.
#'   \item `FolderScripts` – Directory containing R scripts.
#'   \item `FolderReports` – Folder for generated reports.
#'   \item `FolderUtils` – Folder for utility scripts or JAR files.
#'   \item `FolderPython` – Directory for Python-related files.
#'   \item `FolderGlobal` – Folder for global experiment results.
#'   \item `FolderDataset` – Dataset folder.
#'   \item `FolderDatasetX` – Dataset-specific folder.
#'   \item `FolderCV`, `FolderCVTR`, `FolderCVTS`, `FolderCVVL` – Cross-validation folders.
#'   \item `FolderLabelSpace` – Folder for label space representations.
#'   \item `FolderNamesLabels` – Folder containing label name mappings.
#' }
#'
#' @examples
#' \dontrun{
#' parameters <- list(
#'   Config.File = list(Folder.Results = "/tmp/results"),
#'   dataset_name = "example_dataset"
#' )
#' dirs <- directories(parameters)
#'
#' # Access one of the returned folders:
#' dirs$FolderResults
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [dir.create()] and [dir.exists()] for folder management,
#' [setwd()] for setting working directories.
#'
#' @note
#' - All paths are constructed relative to `FolderRoot`, which should be
#'   defined in the global environment before calling this function.
#' - The function assumes that `dataset_name` and `folderResults`
#'   variables are already set and valid.
#' - Existing folders are not overwritten; only missing ones are created.
#'
#' @export
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
#' Extract dataset metadata into a structured list
#'
#' @description
#' This function organizes and extracts key information from a dataset object
#' into a structured list containing descriptive statistics and metadata fields.
#' It is primarily used to standardize dataset information for later processing
#' in multilabel learning experiments.
#'
#' @param dataset A list or data structure containing metadata and statistics
#' for a given dataset. It must include named elements such as:
#' `ID`, `Name`, `Instances`, `Inputs`, `Labels`, `LabelsSets`, `Single`,
#' `MaxFreq`, `Card`, `Dens`, `Mean`, `Scumble`, `TCS`, `AttStart`, `AttEnd`,
#' `LabelStart`, and `LabelEnd`.
#'
#' @return
#' A list containing the extracted information from the input dataset, with
#' fields for identification, size, label statistics, and structural boundaries.
#'
#' The returned list includes:
#' \itemize{
#'   \item \code{id} — Dataset identifier.
#'   \item \code{name} — Dataset name.
#'   \item \code{instances} — Number of instances.
#'   \item \code{inputs} — Number of input features.
#'   \item \code{labels} — Number of output labels.
#'   \item \code{LabelsSets} — Label set definitions.
#'   \item \code{single} — Indicator for single-label datasets.
#'   \item \code{maxfreq} — Maximum label frequency.
#'   \item \code{card}, \code{dens}, \code{mean}, \code{scumble}, \code{tcs} — Statistical measures.
#'   \item \code{attStart}, \code{attEnd} — Input attribute index range.
#'   \item \code{labStart}, \code{labEnd} — Label attribute index range.
#' }
#'
#' @examples
#' \dontrun{
#' dataset <- list(
#'   ID = 1, Name = "Example", Instances = 100, Inputs = 10, Labels = 3,
#'   LabelsSets = list(c(1, 0, 1)), Single = FALSE, MaxFreq = 0.6,
#'   Card = 1.2, Dens = 0.4, Mean = 0.5, Scumble = 0.1, TCS = 0.9,
#'   AttStart = 1, AttEnd = 10, LabelStart = 11, LabelEnd = 13
#' )
#' info <- infoDataSet(dataset)
#' print(info$name)
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' Functions that organize and preprocess multilabel datasets.
#'
#' @note
#' The input `dataset` structure must contain all required fields with consistent naming.
#'
#' @export
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
#' Generate multilabel confusion matrices and related statistics
#'
#' @description
#' This function computes confusion matrix elements (True Positives, False Positives,
#' True Negatives, and False Negatives) for each label in a multilabel classification task.
#' It also saves several CSV reports containing summary statistics and intermediate results
#' for further analysis.
#'
#' @details
#' For each label, the function calculates:
#' \itemize{
#'   \item The number of positive and negative instances.
#'   \item Totals of true and predicted positive/negative cases.
#'   \item Element-wise confusion values (TP, FP, FN, TN).
#'   \item Aggregated confusion matrices per label.
#' }
#' The results are stored in CSV files, organized under the directory specified in `salva`.
#'
#' @param true A binary matrix or data frame with the ground truth labels (0 or 1).
#' Each column corresponds to one label.
#' @param pred A binary matrix or data frame with predicted labels (0 or 1),
#' with the same dimensions and order as `true`.
#' @param type Character. A short string used as a prefix for naming the output files
#' (e.g., `"train"`, `"test"`, `"validation"`).
#' @param salva Character. The directory path where CSV result files will be saved.
#' @param nomes.rotulos Character vector containing the names of the labels (used as row names in the outputs).
#'
#' @return
#' This function does not return a value. It writes multiple CSV files to disk:
#' \itemize{
#'   \item `*-ins-pn.csv`: number of positive and negative instances per label.
#'   \item `*-trues-preds.csv`: total counts of true/predicted positives and negatives.
#'   \item `*-tfpn.csv`: element-wise TP, FP, FN, TN matrices.
#'   \item `*-matrix-confusion.csv`: aggregated confusion matrix per label.
#' }
#'
#' @examples
#' \dontrun{
#' true <- data.frame(L1 = c(1,0,1,0), L2 = c(0,1,1,0))
#' pred <- data.frame(L1 = c(1,0,0,0), L2 = c(1,1,0,0))
#' nomes.rotulos <- c("Label1", "Label2")
#' output_dir <- "results"
#' dir.create(output_dir, showWarnings = FALSE)
#'
#' matrix.confusao(true, pred, type = "test", salva = output_dir, nomes.rotulos)
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [write.csv()], [data.frame()], and other functions for managing multilabel results.
#'
#' @note
#' Ensure that `true` and `pred` have identical dimensions and column order.
#' The directory specified in `salva` must exist or be writable.
#'
#' @export
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


#########################################################################################################
#' Evaluate multilabel classification results and save performance metrics
#'
#' @description
#' This function performs multilabel model evaluation by computing confusion matrices
#' and derived performance measures. It saves the main evaluation results to CSV files
#' for further analysis and reporting.
#'
#' @details
#' The function uses `multilabel_confusion_matrix()` to generate per-label confusion
#' matrices from the true and predicted multilabel sets. Then, it computes overall
#' evaluation metrics using `multilabel_evaluate()` and organizes the results in
#' structured tables. Summary information, including true/false positives and negatives,
#' is saved in CSV format.
#'
#' @param f Integer or character. Identifier for the current fold (used in cross-validation).
#' It is appended to column names in the output.
#' @param y_true Data frame or list. Ground truth labels for the multilabel task.
#' Must contain one column per label.
#' @param y_pred Data frame or list. Predicted labels with the same structure as `y_true`.
#' @param salva Character. Directory path where result files will be saved.
#' @param nome Character. Base name used to name the output CSV files.
#'
#' @return
#' This function writes the following files to disk:
#' \itemize{
#'   \item `<nome>.csv` — evaluation metrics for the given fold.
#'   \item `<nome>-utiml.csv` — detailed confusion matrix statistics (optional, currently commented).
#' }
#' The function does not return an object in R (invisible `NULL`).
#'
#' @examples
#' \dontrun{
#' # Example: evaluating multilabel predictions for one fold
#' y_true <- data.frame(L1 = c(1,0,1,0), L2 = c(0,1,1,0))
#' y_pred <- data.frame(L1 = c(1,0,0,0), L2 = c(1,1,0,0))
#' output_dir <- "results"
#' dir.create(output_dir, showWarnings = FALSE)
#'
#' avaliacao(f = 1, y_true = y_true, y_pred = y_pred,
#'           salva = output_dir, nome = "Fold1_results")
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [multilabel_confusion_matrix()], [multilabel_evaluate()],
#' [write.csv()] for saving structured outputs.
#'
#' @note
#' The helper functions `multilabel_confusion_matrix()` and `multilabel_evaluate()`
#' must be available in the environment or loaded from the appropriate library.
#'
#' @export
avaliacao <- function(f, y_true, y_pred, salva, nome){
  
  #salva.0 = paste(salva, "/", nome, "-conf-mat.txt", sep="")
  #sink(file=salva.0, type="output")
  confmat = multilabel_confusion_matrix(y_true, y_pred)
  #print(confmat)
  #sink()
  
  resConfMat = multilabel_evaluate(confmat)
  resConfMat = data.frame(resConfMat)
  names(resConfMat) = paste("Fold-", f, sep="")
  Measure = rownames(resConfMat)
  resConfMat = data.frame(Measure, resConfMat)
  rownames(resConfMat) = NULL
  salva.1 = paste(salva, "/", nome, ".csv", sep="")
  write.csv(resConfMat, salva.1, row.names = FALSE)
  
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


#########################################################################################################
#' Compute and export ROC curve evaluation for multilabel classification
#'
#' @description
#' This function evaluates the ROC (Receiver Operating Characteristic) metrics
#' for multilabel classification results and exports the computed metrics to a CSV file.
#' Optionally, the function can also plot and save the ROC curve (the plotting code
#' is currently commented out but preserved for reference).
#'
#' @details
#' The function uses \code{mldr_evaluate()} to compute performance metrics and
#' ROC-related statistics for multilabel models. The results are converted into a
#' clean data frame and saved to disk. If the ROC object is available, its AUC
#' (Area Under the Curve) is extracted and appended to the output.
#'
#' @param f Integer or character. Identifier of the fold being evaluated (used in cross-validation).
#' @param y_pred Data frame or list. Predicted label scores or probabilities from the model.
#' @param test Data frame or list. True labels for the test partition.
#' @param Folder Character. Directory path where output files (e.g., plots or CSVs) will be saved.
#' @param nome Character. The name of the output CSV file (including path if needed).
#'
#' @return
#' A CSV file is written to disk containing all evaluation metrics derived from
#' \code{mldr_evaluate()}, including (if available) the ROC AUC value.
#' The function does not return an R object (invisible \code{NULL}).
#'
#' @examples
#' \dontrun{
#' test <- data.frame(L1 = c(1, 0, 1, 0), L2 = c(0, 1, 1, 0))
#' y_pred <- data.frame(L1 = c(0.9, 0.2, 0.8, 0.3),
#'                      L2 = c(0.1, 0.7, 0.6, 0.4))
#' output_dir <- "results"
#' dir.create(output_dir, showWarnings = FALSE)
#'
#' roc.curve(f = 1, y_pred = y_pred, test = test,
#'           Folder = output_dir,
#'           nome = paste0(output_dir, "/fold1_roc.csv"))
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [mldr_evaluate()] for multilabel evaluation,
#' [plot()] for ROC curve visualization,
#' and [write.csv()] for saving structured metrics.
#'
#' @note
#' Ensure that the \code{mldr} package (or any library providing \code{mldr_evaluate})
#' is loaded in your environment. The commented ROC plotting section can be re-enabled
#' if graphical outputs are required.
#'
#' @export
roc.curve <- function(f, y_pred, test, Folder, nome){
  
  res = mldr_evaluate(test, y_pred)
  
  ###############################################################
  # PLOTANDO ROC CURVE
  #name = paste(Folder, "/roc.pdf", sep="")
  #pdf(name, width = 10, height = 8)
  #print(plot(res$roc, print.thres = 'best', print.auc=TRUE, 
  #            print.thres.cex=0.7, grid = TRUE, identity=TRUE,
  #            axes = TRUE, legacy.axes = TRUE, 
  #            identity.col = "#a91e0e", col = "#1161d5",
  #            main = paste("fold ", f, " ", nome, sep="")))
  #dev.off()
  #cat("\n")
  
  ###############################################################
  # Transformar a lista em data frame, removendo 'roc' para evitar problemas
  df_res <- res
  if("roc" %in% names(df_res)) df_res$roc <- NULL
  
  df_metrics <- data.frame(
    metric = names(df_res),
    value = unlist(df_res)
  )
  
  # Se quiser, também adiciona a AUC do objeto ROC
  if(!is.null(res$roc)) {
    df_metrics <- rbind(df_metrics, data.frame(
      metric = "roc_auc",
      value = res$roc$auc
    ))
  }
  
  colnames(df_metrics) = c("Measure", "Value")
  write.csv(df_metrics, nome, row.names = FALSE)
  
}


#' Compute and export AUPRC (Precision-Recall) metrics for multilabel classification
#'
#' @description
#' This function computes the AUPRC (Area Under the Precision-Recall Curve)
#' for each label in a multilabel classification problem. It also calculates
#' macro and micro AUPRC scores and exports the results as CSV files.
#' Optional plotting code for PR curves is included (commented out).
#'
#' @details
#' The function evaluates per-label and aggregated AUPRC metrics using
#' the \code{PRROC} package. For each label, a precision-recall curve is
#' generated when possible (skipping labels with only one class present).
#' It writes two CSV outputs:
#' \itemize{
#'   \item \code{r-auprc-per-label.csv}: AUPRC values for each label.
#'   \item A file specified by \code{nome}: macro and micro AUPRC scores.
#' }
#'
#' @param y_true Matrix or data frame. True binary labels (0 or 1) for each class.
#' @param y_proba Matrix or data frame. Predicted probabilities or confidence scores for each class.
#' @param Folder Character. Directory where output CSV files will be saved.
#' @param nome Character. The name of the main output CSV file containing macro and micro AUPRC values.
#'
#' @return
#' Two CSV files are written to disk:
#' \enumerate{
#'   \item \code{r-auprc-per-label.csv}: per-label AUPRC values.
#'   \item The file specified in \code{nome}: overall macro and micro AUPRC values.
#' }
#' The function does not return an R object (invisible \code{NULL}).
#'
#' @examples
#' \dontrun{
#' # Example data
#' y_true <- data.frame(
#'   L1 = c(1, 0, 1, 0),
#'   L2 = c(0, 1, 1, 0)
#' )
#' y_proba <- data.frame(
#'   L1 = c(0.9, 0.2, 0.8, 0.3),
#'   L2 = c(0.1, 0.7, 0.6, 0.4)
#' )
#'
#' # Output directory and filenames
#' Folder <- "results"
#' dir.create(Folder, showWarnings = FALSE)
#'
#' auprc.curve(y_true = y_true, y_proba = y_proba,
#'             Folder = Folder,
#'             nome = paste0(Folder, "/auprc-summary.csv"))
#' }
#'
#' @author Elaine Cecília Gatto - Cissa
#'
#' @seealso
#' [PRROC::pr.curve()] for PR curve and AUPRC computation,
#' [write.csv()] for saving structured metrics.
#'
#' @note
#' This function requires the \code{PRROC} package.
#' Labels with no positive or negative instances are skipped (AUPRC = NA).
#' The commented plotting code can be re-enabled to generate per-label
#' and global PR curve visualizations.
#'
#' @export
auprc.curve <- function(y_true, y_proba, Folder, nome){
  library(PRROC)
  
  # Garantindo que y_true e y_score sejam matrizes
  y_true <- as.matrix(y_true)
  y_score <- as.matrix(y_proba)
  
  auprc_list <- c()
  
  for(i in 1:ncol(y_true)){
    cat("\n", i)
    # Evita erro quando não houver positivos ou negativos
    if(sum(y_true[, i] == 1) == 0 | sum(y_true[, i] == 0) == 0) {
      auprc_list[i] <- NA
      next
    }
    
    pr_obj <- pr.curve(
      scores.class0 = y_score[y_true[, i] == 1, i],
      scores.class1 = y_score[y_true[, i] == 0, i],
      curve = TRUE
    )
    
    auprc_list[i] <- pr_obj$auc.integral
    
    #nome = paste("AUPRC-Label", i, ".pdf", sep="")
    #pdf(file = paste0(Folder, "/", nome), width = 8, height = 6)
    #plot(pr_obj, main = paste("PR Curve label", i))
    #dev.off()
  }
  
  auprc_per_labels = data.frame(t(auprc_list))
  colnames(auprc_per_labels) = colnames(y_true)
  nome1 = paste(Folder, "/r-auprc-per-label.csv", sep="")
  write.csv(auprc_per_labels, nome1, row.names = FALSE)
  
  # Macro AUPRC
  auprc_macro <- mean(auprc_list, na.rm = TRUE)
  
  # Micro AUPRC: achata tudo
  y_true_vec <- as.vector(y_true)
  y_score_vec <- as.vector(y_score)
  pr_micro <- pr.curve(
    scores.class0 = y_score_vec[y_true_vec == 1],
    scores.class1 = y_score_vec[y_true_vec == 0],
    curve = TRUE
  )
  auprc_micro <- pr_micro$auc.integral
  
  auprc = data.frame(auprc_micro, auprc_macro)
  auprc = data.frame(t(auprc))
  Measure = rownames(auprc)
  auprc = data.frame(Measure, auprc)
  rownames(auprc) = NULL
  colnames(auprc) = c("Measure", "Value")
  write.csv(auprc, nome, row.names = FALSE)
  
  # Salvar gráfico
  # pdf("PR_micro.pdf", width = 8, height = 6)
  # plot(pr_micro, main = "Micro-PR Curve (AUPRC Global)")
  # dev.off()
  
}




###############################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                #
# Thank you very much!                                                        #
###############################################################################
