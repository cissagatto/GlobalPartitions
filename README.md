# Global Partitions for Multilabel Classification

This repository is part of my PhD research at PPG-CC-DC-UFSCar, in collaboration with Katholieke Universiteit Leuven Campus Kulak Kortrijk, Belgium. The goal is to build and test global partitions for multilabel classification. 📚

## How to Cite 📑
If you use this code in your research, please cite the following:

```bibtex
@misc{Gatto2025,
  author = {Gatto, E. C.},
  title = {Global Partitions for Multilabel Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cissagatto/GlobalPartitions}}
}
```

## Source Code Overview 💻
This project is primarily implemented in R and designed to be run in RStudio IDE. It includes several essential scripts:

1. `libraries.R`
2. `utils.R`
3. `global-clus.R`
4. `global-mulan.R`
5. `global-utiml.R`
6. `global-rf.R`
7. `run-clus.R`
8. `run-mulan.R`
9. `run-utiml.R`
10. `run-rf.R`
11. `global.R`
12. `config-files.R`
13. `jobs.R`

**Note:** Random Forest is used for all global versions, except for CLUS (which is a PCT model).  
`global-mulan` and `global-utiml` are not yet implemented. 🔧

## Preparing Your Experiment 🛠️

### STEP 1: Dataset Setup
A file called `datasets-original.csv` should be placed in the **root project folder**. This file contains details for 90 multilabel datasets used in the code. To add a new dataset, include the following information in the file:

| Parameter     | Status    | Description                                          |
|---------------|-----------|------------------------------------------------------|
| Id            | Mandatory | A unique integer identifier for the dataset          |
| Name          | Mandatory | Dataset name (following the benchmark)               |
| Domain        | Optional  | The domain of the dataset                            |
| Instances     | Mandatory | The total number of instances in the dataset         |
| Attributes    | Mandatory | The total number of attributes in the dataset        |
| Labels        | Mandatory | The number of labels in the dataset                  |
| Inputs        | Mandatory | The number of input attributes                       |
| Cardinality   | Optional  | See reference link for more details                  |
| Density       | Optional  | See reference link for more details                  |
| Labelsets     | Optional  | See reference link for more details                  |
| Single        | Optional  | See reference link for more details                  |
| Max.freq      | Optional  | See reference link for more details                  |
| Mean.IR       | Optional  | See reference link for more details                  |
| Scumble       | Optional  | See reference link for more details                  |
| TCS           | Optional  | See reference link for more details                  |
| AttStart      | Mandatory | Column number where the attribute space begins      |
| AttEnd        | Mandatory | Column number where the attribute space ends        |
| LabelStart    | Mandatory | Column number where the label space begins          |
| LabelEnd      | Mandatory | Column number where the label space ends            |
| Distinct      | Optional  | See reference link for more details                  |
| xn            | Mandatory | Kohonen map dimension X                             |
| yn            | Mandatory | Kohonen map dimension Y                             |
| gridn         | Mandatory | The product of X and Y for Kohonen's map (must be square) |
| max.neighbors | Mandatory | The maximum number of neighbors is `Labels - 1`      |

For detailed explanations of each property, [click here](https://link.springer.com/book/10.1007/978-3-319-41111-8).

### STEP 2: Cross-Validation Files
The experiment requires pre-processed cross-validation files in `.tar.gz` format. You can download the 10-fold files for multilabel datasets [here](https://1drv.ms/u/s!Aq6SGcf6js1mrZJSkZ3VEJ217rEd5A?e=IH73m3).

For new datasets, you can generate these files by following the instructions in [this repository](https://github.com/cissagatto/crossvalidationmultilabel). After generating the files, place the `.tar.gz` archive in any directory, and provide the absolute path in the configuration file for the `global.R` script.

### STEP 3: Required Libraries 📦
Ensure that all necessary Java, R, and Python libraries are installed on your system. This code does not automatically install packages! 🚨

You can use the [Conda Environment ELCC](https://1drv.ms/u/s!Aq6SGcf6js1mw4hbhU9Raqarl8bH8Q?e=IA2aQs) that I created to perform this experiment. Below are the links to download the files. Try to use the command below to extract the environment to your computer:

```
conda env create -f ELCC.yml
```

For more information on Mini-Forge-Conda environments, refer to the [official documentation]().

Alternatively, you can run the code using an AppTainer container. Check the [tutorial](https://rpubs.com/cissagatto/apptainer-slurm-r) for setup instructions (in Portuguese).

### STEP 4: Configuration File ⚙️
You will need a `.csv` configuration file with the following fields:

| Config           | Value                                                                    |
|------------------|--------------------------------------------------------------------------|
| FolderScripts    | Absolute path to the R folder scripts                                    |
| Dataset_Path     | Absolute path to the folder where the dataset's `.tar.gz` file is stored |
| Temporary_Path   | Absolute path to the folder for temporary processing                     |
| Implementation   | Choose from "clus", "mulan", "rf", or "utiml"                            |
| Dataset_Name     | Dataset name (from `datasets-original.csv`)                              |
| Number_Dataset   | Dataset number (from `datasets-original.csv`)                            |
| Number_Folds     | Number of folds for cross-validation                                     |
| Number_Cores     | Number of cores for parallel processing                                  |

We recommend using directories like `/dev/shm`, `/tmp`, or `/scratch` for temporary storage.

For detailed instructions on configuration, refer to the example files.

## Software Requirements 💻
- **RStudio** Version 1.4.1106
- **R Language** Version 4.5.1
- **Python** Version 3.10.6

IMPORTANT: there are some versions of python that the AUPRC/ROC computation does not work well.

## Hardware Requirements 🖥️
This code can run in parallel, and it's highly recommended to use parallel processing. You can configure the number of cores via the command line. In our experiments, we used 10 cores. To ensure reproducibility, we suggest using the same configuration.

The code was tested on the following machine:

*System:*

Host: bionote | Kernel: 5.8.0-53-generic | x86_64 bits: 64 | Desktop: Gnome 3.36.7 | Distro: Ubuntu 20.04.2 LTS (Focal Fossa)

*CPU:*

Topology: 6-Core | model: Intel Core i7-10750H | bits: 64 | type: MT MCP | L2 cache: 12.0 MiB | Speed: 800 MHz | min/max: 800/5000 MHz Core speeds (MHz): | 1: 800 | 2: 800 | 3: 800 | 4: 800 | 5: 800 | 6: 800 | 7: 800 | 8: 800 | 9: 800 | 10: 800 | 11: 800 | 12: 800 |

For cluster execution, we used a UFSCar cluster.

## Running the Code 🚀
To execute the code, open the terminal, navigate to the `/Global-Partitions/R/` directory, and run:

```
Rscript global.R [absolute_path_to_config_file]
```

For example:

```
Rscript global.R "~/Global-Partitions/config-files/gr-emotions.csv"
```

## Results 📊
Results will be stored in the `REPORTS` folder in the root directory.

## Download Results ⬇️
Coming soon!

## Acknowledgments 🙏
- This study was supported by CAPES (Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil) - Finance Code 001.
- Funded in part by CNPQ (Conselho Nacional de Desenvolvimento Científico e Tecnológico - Brasil), Process number 200371/2022-3.
- Special thanks to FAPESP (Fundação de Amparo à Pesquisa do Estado de São Paulo) for financial support.

## Contact 📧
Feel free to reach out: [elainececiliagatto@gmail.com](mailto:elainececiliagatto@gmail.com)

## Links 🌐

| [Site](https://sites.google.com/view/professor-cissa-gatto) | [Post-Graduate Program in Computer Science](http://ppgcc.dc.ufscar.br/pt-br) | [Computer Department](https://site.dc.ufscar.br/) |  [Biomal](http://www.biomal.ufscar.br/) | [CNPQ](https://www.gov.br/cnpq/pt-br) | [Ku Leuven](https://kulak.kuleuven.be/) | [Embarcados](https://www.embarcados.com.br/author/cissa/) | [Read Prensa](https://prensa.li/@cissa.gatto/) | [Linkedin Company](https://www.linkedin.com/company/27241216) | [Linkedin Profile](https://www.linkedin.com/in/elainececiliagatto/) | [Instagram](https://www.instagram.com/cissagatto) | [Facebook](https://www.facebook.com/cissagatto) | [Twitter](https://twitter.com/cissagatto) | [Twitch](https://www.twitch.tv/cissagatto) | [Youtube](https://www.youtube.com/CissaGatto) |

## Thanks! ❤️
