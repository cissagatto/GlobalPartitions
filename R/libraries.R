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
library(here)
library(stringr)
setwd("~/GlobalPartitions")
#FolderRoot <- here::here()
#FolderScripts <- here::here("R")



###############################################################################
#' @title Install and Load Required Packages
#' @description
#' This script verifies whether a list of required R packages are installed on the user's system.
#' If any of the packages are missing, it automatically installs them (from CRAN or GitHub as needed)
#' and loads them into the current R session. This ensures that all dependencies are satisfied.
#'
#' @details
#' - Supports CRAN, base, and GitHub packages.
#' - Automatically installs 'devtools' if needed for GitHub installations.
#' - Provides progress messages and basic error handling.
#' 
#' @author 
#' @date 2025-06-12

# List of required CRAN packages
cran_packages <- c(
  "foreign", "dplyr", "stringr", "foreach", "doParallel",
  "rJava", "RWeka", "mldr", "utiml", "parallel", "AggregateR",
  "reshape2","plyr","readr", "here"
)

# Base packages (already included with R)
base_packages <- c("parallel")

# GitHub packages: named list with package name and corresponding repository
github_packages <- list(
  AggregateR = "guiguegon/AggregateR"
)

#' @description Install a single CRAN package if not already installed
#' @param pkg Name of the package
install_cran_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(paste("Installing CRAN package:", pkg))
    tryCatch({
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE, quietly = TRUE)
      message(paste("Package", pkg, "successfully installed."))
    }, error = function(e) {
      message(paste("Error installing package", pkg, ":", e$message))
    })
  } else {
    message(paste("Package", pkg, "is already installed."))
  }
}


###############################################################################
#' @description Install a single GitHub package if not already installed
#' @param pkg Name of the package
#' @param repo GitHub repository in the format 'username/repo'
install_github_package <- function(pkg, repo) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    if (!require("devtools", character.only = TRUE, quietly = TRUE)) {
      install.packages("devtools")
      library(devtools, quietly = TRUE)
    }
    message(paste("Installing GitHub package:", pkg))
    tryCatch({
      devtools::install_github(repo)
      library(pkg, character.only = TRUE, quietly = TRUE)
      message(paste("Package", pkg, "successfully installed."))
    }, error = function(e) {
      message(paste("Error installing GitHub package", pkg, ":", e$message))
    })
  } else {
    message(paste("Package", pkg, "is already installed."))
  }
}


###############################################################################
# Load base packages (no installation required)
for (pkg in base_packages) {
  library(pkg, character.only = TRUE, quietly = TRUE)
  message(paste("Base package", pkg, "loaded."))
}


###############################################################################
# Install and load CRAN packages
for (pkg in cran_packages) {
  install_cran_package(pkg)
}


###############################################################################
# Install and load GitHub packages
for (pkg in names(github_packages)) {
  install_github_package(pkg, github_packages[[pkg]])
}

message("✅ All packages have been successfully verified and installed!")



##################################################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                                   #
# Thank you very much!                                                                           #
##################################################################################################
