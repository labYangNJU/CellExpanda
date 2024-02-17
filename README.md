# CellExpanda
# Introduction
CellExpanda is implemented by python to take full advantage of paired single-cell multimodal data for holistic representation of cells.


<img width="641" alt="Screen Shot 2024-01-25 at 5 09 08 PM" src="https://github.com/labYangNJU/CellExpanda/assets/80734679/25ee5344-a7d1-4c0f-b54f-9acf0e194a9b">


For more details, please check our paper: Unlocking full potential of paired single-cell multi-omics to expand cellular view with CellExpanda.


# Input Data
CellExpanda takes count matrices from paired single-cell multimodal data. There is no limitation for the types and numbers of modalities.
An expample input dataset can be found in the example/ directory. Note: You should change the "sampleName" of files according to your own dataset.
Totally, there are four files required:

1.sampleName_SparseMatrix.txt  # The raw count sparsematrix with three columns.

2.sampleName_barcode.txt  # The barcode file with one barcode per line.

3.feature_discriminator.txt  # The file with feature name and weight (0 indicated with no weight and 1 indicated with weight).



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
