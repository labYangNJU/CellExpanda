# CellExpanda
# Introduction
CellExpanda is implemented by python to take full advantage of paired single-cell multimodal data for holistic representation of cells. Running CellExpanda on CUDA is recommended if available.


<img width="794" alt="截屏2024-02-17 12 49 59" src="https://github.com/labYangNJU/CellExpanda/assets/80734679/67f27cfd-621b-4fd6-81ec-cf9854baf72a">


For more details, please check our paper: Unlocking full potential of paired single-cell multi-omics to expand cellular view with CellExpanda.


# Directory structure
      .
      ├── CellExpands          # Main Python package
            ├── configs        # Config yaml file 
            ├── main.py
            ├── model.py
            ├── modules.py
            ├── ...  
      ├── data                 # Datasets
      ├── scripts              # Scripts for reproducibility of results in the manuscript
      ├── env.yaml             # Reproducible Python environment via conda
      ├── LICENSE
      └── README.md



# Input Data
CellExpanda takes count matrices from paired single-cell multimodal data. There is no limitation for the types and numbers of modalities.
An expample input dataset can be found in the example/ directory. Note: You should change the "sampleName" of files according to your own dataset.
Totally, there are three files required:

1.sampleName_SparseMatrix.txt 

The raw count sparse matrix for the multimodal dataset with features from all modalities.

2.sampleName_barcode.txt  

The barcode file with one barcode per line for each cell.

3.feature_discriminator.txt  

The file with feature information. You can increase weight for selected features which could be more important for cell clustering. For example, features can be selected using tools like DubStepR (PMID: 34615861
        
        
        
        ). The first column is feature name and the second column is the additional weight for particular features. If no additonal weigh is needed, set 0 for the feature.



# Installation
1. conda env create -f CellExpanda.yaml
2. install the following python packages by pip : sklearn; torch; matplotlib; numpy; networkx; igraph; pyyaml; pandas; tensorboardX; python-louvain; umap; torchsummary; pysam; pytorch_metric_learning.
3. conda install conda-forge::leidenalg


# Usage
1.Activate conda environment

conda activate CellExpanda

2.Configure your sampleName.yaml files under the configs/ directory. Detailed instructions can be found in the ReadMe_for_yaml_file.txt file.

3.Run the CellExpanda model.

python3 main.py --dataset=sampleName


# Dependencies
+ Python3.6
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
+ umap
+ torchsummary
+ pysam
+ pytorch_metric_learning
