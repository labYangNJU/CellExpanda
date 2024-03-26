# CellMATE
# Introduction
CellMATE is implemented by python to take full advantage of paired single-cell multimodal data for holistic representation of cells. Running CellMATE on GPU is recommended if available.


# Directory structure
      .
      ├── CellMATE          # Main Python package
            ├── configs        # Config yaml files for each dataset 
            ├── result         # result files for each dataset 
            ├── main.py
            ├── model.py
            ├── modules.py
            ├── ...  
      ├── data                 # Input datasets
      ├── scripts              # Scripts for reproducibility of results in the manuscript
      ├── env.yaml             # Reproducible Python environment via conda
      ├── LICENSE
      └── README.md


# Input Data
CellMATE takes count matrices from paired single-cell multimodal data. There is no limitation for the types and numbers of modalities.
An expample input dataset can be found in the data/ directory. Note: You should change the "sampleName" of files according to your own dataset.
Three files are required:

1.sampleName_SparseMatrix.txt 

The raw count sparse matrix for the multimodal dataset with features from all modalities.

2.sampleName_barcode.txt  

The barcode file with one barcode per line for each cell.

3.feature_discriminator.txt  

The file with feature information with feature name and the additional weight for each feature. 

(Optional) You can increase weight for selected features which could be more important for cell clustering. For example, features can be selected using tools like DubStepR (PMID: 34615861). If no additonal weigh is needed, set 0 for the feature.


# Installation

option 1:
Create an environment from the env.yml file (We tested with conda==4.12.0, python==3.6.9).

      conda env create -f env.yaml

option 2:
Create your own conda environment with python3 and install all the dependencies.

Notes: 
If any dependencies are not installed automatically, you should install them by pip or conda.
If GPU is used, the torch version should be compiled with your version of the CUDA driver.


# Usage
1.Activate conda environment

      conda activate CellMATE

2.Configure your sampleName.yaml files under the directory CellMATE/configs/. Detailed instructions can be found in the ReadMe_for_ConfigYaml_file.txt file.

3.Run the CellMATE model.

      python3 main.py --dataset=sampleName --mode=train


# Output 
The output includes 1) the representation of cells; 2) the reconstructed single-cell multimodal data.

The representation of cells would be generated in the directory CellMATE/result/.

      result
      ├── sampleName                                
            ├── latent-sampleName-512-10.txt        # The latent representation of cells (can be used to generate UMAP emmbeddings).
            ├── CellMATE-sampleName-tsne-pred.png       # The tSNE visualization of cells. 
            ├── ...  
      └── sampleName-512-10-cluster_result.csv      # The clustering information of cells with tSNE/UMAP emmbeddings. 
      
The reconstructed data would be generated in the directory CellMATE/result/sampleName and can be extracted as: 

      python3 main.py --dataset=sampleName --mode=reconstruct


# Example
An example we shown here is the dataset joint profiling scChIP of three distinct histone modifications (MulTI-Tag), which is used in our paper.

The dataset can be downloaded from the Zenodo repository: https://doi.org/10.5281/zenodo.6636675. 

Step 1. Prepare the input dataset.

The input dataset can be prepared using the Data_prepare_for_scChip_K27_K36_K4m1_dataset.R script under the directory scripts/.

Step 2. Run CellMATE with the config file under the directory CellMATE/configs/.

      python3 main.py --dataset=scChip_K27_K36_K4m1 --mode=train

Step 3. You can check the output with the one provided in the directory CellMATE/result/scChip_K27_K36_K4m1_out/.

Note: The results may be a little different due to the differences in versions of dependencies.


# Dependencies
+ Python3
+ sklearn
+ torch
+ matplotlib
+ numpy
+ networkx
+ igraph
+ pyyaml
+ pandas
+ tensorboardX
+ python-louvain
+ leidenalg
+ umap-learn
+ torchsummary
+ pysam
+ pytorch_metric_learning
