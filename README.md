# CellExpanda
# Introduction
CellExpanda is implemented by python to take full advantage of paired single-cell multimodal data for holistic representation of cells.


<img width="659" alt="Screen Shot 2024-01-25 at 3 05 53 PM" src="https://github.com/labYangNJU/CellExpanda/assets/80734679/754c85f1-aaea-4f3b-a0bd-9e49176bde9d">


For more details, please check our paper: Taking full advantage of paired single-cell multimodal data for holistic representation of cells with CellExpanda.


# Input Data
CellExpanda takes count matrices from paired single-cell multimodal data consisting of different types and numbers of modalities.
An expample input dataset can be found in the example/ directory. Note: You should name each file the same format as shown in the example.
Totally, there are four files required:

1.sampleName_SparseMatrix.txt  # The raw count sparsematrix with three columns.

2.sampleName_barcode.txt  # The barcode file with one barcode per line.

3.sampleName_celltype_info.csv  # The cell-type label file with one barcode and cell-type label per line. (header is required!)

4.feature_discriminator.txt  # The file with feature name and weight (o indicated with no weight and 1 indicated with weight).



# Installation
conda env create -f CellExpanda.yaml


# Usage
1.Activate conda environment

conda activate CellExpanda

2.Configure your sampleName.yaml files under the configs/ directory. Detailed instructions can be found in the provided sampleName.yaml file.

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
