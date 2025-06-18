cat("\n\n###############################################################")
  cat("\n# RSCRIPT: START EXECUTE GLOBAL PARTITIONS                    #")
  cat("\n###############################################################\n\n")


# clean
rm(list=ls())


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

#getwd()

cat("\n################################")
cat("\n# Set Work Space               #")
cat("\n###############################\n\n")
library(here)
library(stringr)
FolderRoot <- here::here()



cat("\n########################################")
cat("\n# R Options Configuration              #")
cat("\n########################################\n\n")
options(java.parameters = "-Xmx64g")  # JAVA
options(show.error.messages = TRUE)   # ERROR MESSAGES
options(scipen=20)                    # number of places after the comma


cat("\n########################################")
cat("\n# Creating parameters list              #")
cat("\n########################################\n\n")
parameters = list()


cat("\n########################################")
cat("\n# Reading Datasets-Original.csv        #")
cat("\n########################################\n\n")
setwd(FolderRoot)
datasets <- data.frame(read.csv("datasets-original.csv"))
parameters$Datasets.List = datasets


cat("\n#####################################")
cat("\n# GET ARGUMENTS FROM COMMAND LINE   #")
cat("\n#####################################\n\n")
args <- commandArgs(TRUE)

config_file <- args[1]


# config_file = "~/GlobalPartitions/config-files/gr-flags.csv"

# /home/cissagatto/Documentos/GlobalPartitions/config-files/rf



parameters$Config.File$Name = config_file
if(file.exists(config_file)==FALSE){
  cat("\n################################################################")
  cat("#\n Missing Config File! Verify the following path:              #")
  cat("#\n ", config_file, "                                            #")
  cat("#################################################################\n\n")
  break
} else {
  cat("\n########################################")
  cat("\n# Properly loaded configuration file!  #")
  cat("\n########################################\n\n")
}


cat("\n########################################")
cat("\n# Config File                          #\n")
config = data.frame(read.csv(config_file))
print(config)
cat("\n########################################\n\n")


cat("\n########################################")
cat("\n# Getting Parameters                   #\n")
cat("\n########################################")
FolderScripts = toString(config$Value[1])
FolderScripts = str_remove(FolderScripts, pattern = " ")
parameters$Directories$FolderScripts = FolderScripts

dataset_path = toString(config$Value[2])
dataset_path = str_remove(dataset_path, pattern = " ")
parameters$Config.File$Dataset.Path = dataset_path

folderResults = toString(config$Value[3]) 
folderResults = str_remove(folderResults, pattern = " ")
parameters$Config.File$Folder.Results = folderResults

implementation = toString(config$Value[4])
implementation = str_remove(implementation, pattern = " ")
parameters$Config.File$Implementation = implementation

dataset_name = toString(config$Value[5])
dataset_name = str_remove(dataset_name, pattern = " ")
parameters$Config.File$Dataset.Name = dataset_name

number_dataset = as.numeric(config$Value[6])
parameters$Config.File$Number.Dataset = number_dataset

number_folds = as.numeric(config$Value[7])
parameters$Config.File$Number.Folds = number_folds

number_cores = as.numeric(config$Value[8])
parameters$Config.File$Number.Cores = number_cores

ds = datasets[number_dataset,]
parameters$Dataset.Info = ds

cat("\n########################################")
cat("\n# Loading R Sources                    #")
cat("\n########################################\n\n")
source(file.path(FolderScripts, "libraries.R"))
source(file.path(FolderScripts, "utils.R"))


cat("\n########################################")
cat("\n# Creating temporary processing folder #")
cat("\n########################################\n\n")
if (dir.exists(folderResults) == FALSE) {dir.create(folderResults)}


cat("\n###############################\n")
cat("\n# Get directories             #")
cat("\n###############################\n\n")
diretorios <- directories(parameters)
parameters$Directories = diretorios


cat("\n####################################################################")
cat("\n# Checking the dataset tar.gz file                                 #")
cat("\n####################################################################\n\n")
str00 = paste(dataset_path, "/", parameters$Config.File$Dataset.Name,".tar.gz", sep = "")
str00 = str_remove(str00, pattern = " ")

if(file.exists(str00)==FALSE){
  
  cat("\n######################################################################")
  cat("\n# The tar.gz file for the dataset to be processed does not exist!    #")
  cat("\n# Please pass the path of the tar.gz file in the configuration file! #")
  cat("\n# The path entered was: ", str00, "                                  #")
  cat("\n######################################################################\n\n")
  break
  
} else {
  
  cat("\n####################################################################")
  cat("\n# tar.gz file of the DATASET loaded correctly!                     #")
  cat("\n####################################################################\n\n")
  
  # COPIANDO
  str01 = paste("cp ", str00, " ", parameters$Directories$FolderDataset, sep = "")
  res = system(str01)
  if (res != 0) {
    cat("\nError: ", str01)
    break
  }
  
  # DESCOMPACTANDO
  str02 = paste("tar xzf ",  parameters$Directories$FolderDataset, "/", ds$Name,
                ".tar.gz -C ",  parameters$Directories$FolderDataset, sep = "")
  res = system(str02)
  if (res != 0) {
    cat("\nError: ", str02)
    break
  }
  
  #APAGANDO
  str03 = paste("rm ",  parameters$Directories$FolderDataset, "/", ds$Name,
                ".tar.gz", sep = "")
  res = system(str03)
  if (res != 0) {
    cat("\nError: ", str03)
    break
  }
  
}



##############################################################################
#
##############################################################################
if(implementation=="clus"){
  
  source(file.path(parameters$Directories$FolderScripts, "run-clus.R"))
  
  cat("\n\n############################################################")
  cat("\n# RSCRIPT GLOBAL START                                     #")
  cat("\n############################################################\n\n")
  timeFinal <- system.time(results <- run.clus(parameters))  
  
  
  cat("\n\n#####################################################")
  cat("\n# RSCRIPT SAVE RUNTIME                              #")
  cat("\n#####################################################\n\n")
  result_set <- t(data.matrix(timeFinal))
  setwd(parameters$Directories$FolderGlobal)
  write.csv(result_set, "Final-Runtime.csv")
  
  
  cat("\n\n####################################################")
  cat("\n# RSCRIPT DELETE                                   #")
  cat("\n####################################################\n\n")
  str5 = paste("rm -r ", parameters$Directories$FolderDataset, sep="")
  print(system(str5))
  
} else if(implementation=="rf"){
  
  source(file.path(parameters$Directories$FolderScripts, "run-rf.R"))
  
  cat("\n\n############################################################")
     cat("\n# RSCRIPT GLOBAL RANDOM FORESTS START                     #")
     cat("\n###########################################################\n\n")
  timeFinal <- system.time(results <- run.rf(parameters))  
  
  
  cat("\n\n#####################################################")
  cat("\n# RSCRIPT SAVE RUNTIME                              #")
  cat("\n#####################################################\n\n")
  result_set <- t(data.matrix(timeFinal))
  setwd(parameters$Directories$FolderGlobal)
  write.csv(result_set, "Final-Runtime.csv", row.names = FALSE)
  
  
  cat("\n\n###################################################")
  cat("\n# RSCRIPT DELETE                                  #")
  cat("\n###################################################\n\n")
  str5 = paste("rm -r ", parameters$Directories$FolderDataset, sep="")
  print(system(str5))
  
  
  cat("\n\n###################################################################")
  cat("\n# GLOBAL: COMPRESS RESULTS                                      #")
  cat("\n#####################################################################\n\n")
  tar_file <- paste0(parameters$Directories$FolderResults, "/", 
                     parameters$Dataset.Info$Name,
                     "-results-global.tar.gz")
  
  str_01 <- paste(
    "tar -zcvf", tar_file,
    "-C", parameters$Directories$FolderGlobal, "."
  )
  cat("\nComando:\n", str_01, "\n")
  system(str_01)
  
  
  
  cat("\n\n###################################################################")
  cat("\n# ====> GPC: COPY TO HOME                                     #")
  cat("\n#####################################################################\n\n")
  str_0 = parameters$Directories$FolderReports
  if(dir.exists(str_0)==FALSE){dir.create(str0)}
  
  str_03 = paste(parameters$Directories$FolderResults, "/",
                 parameters$Dataset.Info$Name,
                 "-results-global.tar.gz", sep="")
  
  str_04 = paste("cp ", str_03, " ", str_0, sep="")
  print(system(str_04))
  
  
  # cat("\n\n######################################################")
  # cat("\n# RSCRIPT COPY TO GOOGLE DRIVE                       #")
  # cat("\n######################################################\n\n")
  # origem = parameters$Directories$FolderGlobal
  # destino = paste("nuvem:Global/RandomForests/", dataset_name, sep="")
  # comando = paste("rclone -P copy ", origem, " ", destino, sep="")
  # cat("\n", comando, "\n") 
  # a = print(system(comando))
  # a = as.numeric(a)
  # if(a != 0) {
  #   stop("Erro RCLONE")
  #   quit("yes")
  # }
  # 
  
  # cat("\n\n######################################################")
  # cat("\n# RSCRIPT COPY TO GOOGLE DRIVE                       #")
  # cat("\n######################################################\n\n")
  # origem = diretorios$folderDataset
  # destino = paste("nuvem:Datasets/", dataset_name, sep="")
  # comando = paste("rclone -P copy ", origem, " ", destino, sep="")
  # cat("\n", comando, "\n") 
  # a = print(system(comando))
  # a = as.numeric(a)
  # if(a != 0) {
  #   stop("Erro RCLONE")
  #   quit("yes")
  # }
  
} else if(implementation=="mulan"){
  # work in progress
  
} else {
  # work in progress
  
  
}


cat("\n\n#######################################################")
cat("\n# CLEAN                                               #")
cat("\n#######################################################\n\n")
cat("\nDelete folder \n")
str5 = paste("rm -r ", parameters$Directories$FolderResults, sep="")
print(system(str5))


cat("\n\n################################################################")
cat("\n# RSCRIPT SUCCESSFULLY FINISHED                                #")
cat("\n################################################################\n\n")


rm(list = ls())
gc()

###############################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                #
# Thank you very much!                                                        #
###############################################################################
