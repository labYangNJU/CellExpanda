## The following parameters in the configs/xxx.yaml file are required to change according to the types and numbers of modalities in your dataset. ##


name: The name of your dataset.
Example: 
’sampleName'


data_path: The path of the input dataset. Should be a directory.
Example: 
"../data/sampleName"


binary: Whether each modality should be binarized. We recommend ‘False’ for scRNA and scADT, ‘True’ for scATAC and scChIP. However, choosing ‘True’ or ‘False’ has little impact on the results.
Example:
[False, False] # For datasets with two modalities.
[False, False, False] # For datasets with three modalities.
...


normalize: Whether each modality should be normalized. We recommend ‘False’ for scATAC, scChIP and scRNA, ‘True’ for scADT. However, choosing ‘True’ or ‘False’ has little impact on the results.
Example:
[False, False] # For datasets with two modalities.
[False, False, False] # For datasets with three modalities.
...


log_variational: Whether each modality should be log normalized. We recommend ‘False’ for scATAC, scChIP and scADT, ‘True’ for scRNA. However, choosing ‘True’ or ‘False’ has little impact on the results.
Example:
[False, False] # For datasets with two modalities.
[False, False, False] # For datasets with three modalities.
...


dims: The dimension range of the each modality.
Example:
[[0, 13263],[13263,89119]] # If one dataset has 89119 features in total, and the 1~13263 features belong to the first modality, the 13264~89119 features belong to the second modality.
[[0, 13562],[13562,57615],[57615,78798]] # If one dataset has 78798 features in total, and the 1~13562 features belong to the first modality, the 13563~57615 features belong to the second modality, the 57616~78798 features belong to the third modality.
...


reconstruction_loss: The temporary distribution for each modality. We recommend ‘zinb’ for scATAC and scChIP, ‘nb’ for scRNA and scADT. However, choosing ‘zinb’ or ‘nb’ has little impact on the results.
Example:
["zinb", "nb"] # For datasets with two modalities.
["zinb", "nb”, “nb”] # For datasets with three modalities.
...



min_peaks: The minimum fraction of features that can be detected in a cell. 
Example:
0.05


min_cells: The minimum fraction of cells with null-zero values for each feature. 
Example:
0.01


max_cells: The maximum fraction of cells with null-zero values for each feature. 
Example:
0.95


device: The device to run CellExpanda.
Example:
"cuda:0" # If cuda is not available, please change to "cpu".
