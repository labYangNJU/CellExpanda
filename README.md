# CellExpanda
# Introduction
CellExpanda is implemented by python to take full advantage of paired single-cell multimodal data for holistic representation of cells.


<img width="794" alt="截屏2024-02-17 12 49 59" src="https://github.com/labYangNJU/CellExpanda/assets/80734679/67f27cfd-621b-4fd6-81ec-cf9854baf72a">


For more details, please check our paper: Unlocking full potential of paired single-cell multi-omics to expand cellular view with CellExpanda.


# Input Data
CellExpanda takes count matrices from paired single-cell multimodal data. There is no limitation for the types and numbers of modalities.
An expample input dataset can be found in the example/ directory. Note: You should change the "sampleName" of files according to your own dataset.
Totally, there are three files required:

1.sampleName_SparseMatrix.txt 

The raw count sparse matrix for the multimodal dataset with features from all modalities.

2.sampleName_barcode.txt  

The barcode file with one barcode per line for each cell.

3.feature_discriminator.txt  

The file with feature information. You could add weight for features that you think is particular important for cell type idetification using tools like DubStepR (DUBStepR is a scalable correlation-based feature selection method for accurately clustering single-cell data). The first column is feature name and the second column is the weight that add to particular features.



# Installation
conda env create -f CellExpanda.yaml


# Usage
1.Activate conda environment

conda activate CellExpanda

2.Configure your sampleName.yaml files under the configs/ directory. Detailed instructions can be found in the ReadMe_for_yaml_file.txt file.

3.Create sampleName directory under the result/ directory.

4.Run the CellExpanda model.

python3 main.py --dataset=sampleName


# Dependencies
+ Python3
+ sklearn
+ scipy
+ torch
+ matplotlib
+ numpy
+ tqdm
+ networkx
+ igraph
