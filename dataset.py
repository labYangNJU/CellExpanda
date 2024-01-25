import copy
import logging
import os
import urllib.request
from abc import abstractmethod, ABC
from collections import OrderedDict, defaultdict
#from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Tuple, Union, Optional, Callable

import numpy as np
import scipy.sparse as sp_sparse
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)




#@dataclass
class CellMeasurement:
    name: str  # Name of the attribute Eg: 'X'
    data: Union[np.ndarray, sp_sparse.csr_matrix]  # Data itself: Eg: X
    columns_attr_name: str  # Name of the column names attribute : Eg: 'gene_names'
    columns: Union[np.ndarray, List[str]]  # Column names: Eg: gene_names


class GeneExpressionDataset(Dataset):

    def __init__(self):
        # registers
        self.dataset_versions = set()
        self.gene_attribute_names = set()
        self.cell_attribute_names = set()
        self.cell_categorical_attribute_names = set()
        self.attribute_mappings = defaultdict(list)
        self.cell_measurements_columns = dict()

        # initialize attributes
        self._X = None
        self._batch_indices = None
        self._labels = None
        self.n_batches = None
        self.n_labels = None
        self.gene_names = None
        self.cell_types = None
        self.local_means = None
        self.local_vars = None
        self._norm_X = None
        self._corrupted_X = None

        # attributes that should not be set by initialization methods
        self.protected_attributes = ["X"]

    def __repr__(self) -> str:
        if self.X is None:
            descr = "GeneExpressionDataset object (unpopulated)"
        else:
            descr = "GeneExpressionDataset object with n_cells x nb_genes = {} x {}".format(
                self.nb_cells, self.nb_genes
            )
            attrs = [
                "dataset_versions",
                "gene_attribute_names",
                "cell_attribute_names",
                "cell_categorical_attribute_names",
                "cell_measurements_columns",
            ]
            for attr_name in attrs:
                attr = getattr(self, attr_name)
                if len(attr) == 0:
                    continue
                if type(attr) is set:
                    descr += "\n    {}: {}".format(attr_name, str(list(attr))[1:-1])
                else:
                    descr += "\n    {}: {}".format(attr_name, str(attr))

        return descr

    @property
    def X(self):
        return self._X

    def populate_from_data(
        self,
        X: Union[np.ndarray, sp_sparse.csr_matrix],
        Ys: List[CellMeasurement] = None,
        batch_indices: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        labels: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        gene_names: Union[List[str], np.ndarray] = None,
        cell_types: Union[List[str], np.ndarray] = None,
        cell_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        gene_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        remap_attributes: bool = True,
    ):
        # set the data hidden attribute
        self._X = (
            np.ascontiguousarray(X, dtype=np.float32)
            if isinstance(X, np.ndarray)
            else X
        )
        #print('populate', self.X.shape)
        
        self.initialize_cell_attribute(
            "batch_indices",
            np.asarray(batch_indices[:, 0]).reshape((-1, 1))
            if batch_indices is not None
            else np.zeros((X.shape[0], 1)),
            categorical=True,
        )
        self.initialize_cell_attribute(
            "labels",
            np.asarray(labels).reshape((-1, 1))
            if labels is not None
            else np.zeros((X.shape[0], 1)),
            categorical=True,
        )
        
        self.compute_library_size_batch()

        if gene_names is not None:
            self.initialize_gene_attribute(
                "gene_names", np.char.upper(np.asarray(gene_names, dtype="<U64"))
            )
        if cell_types is not None:
            self.initialize_mapped_attribute(
                "labels", "cell_types", np.asarray(cell_types, dtype="<U128")
            )
        # add dummy cell type, for consistency with labels
        elif labels is None:
            self.initialize_mapped_attribute(
                "labels", "cell_types", np.asarray(["undefined"], dtype="<U128")
            )

        # handle additional attributes
        if cell_attributes_dict:
            for attribute_name, attribute_value in cell_attributes_dict.items():
                self.initialize_cell_attribute(attribute_name, attribute_value)
        if Ys is not None:
            for measurement in Ys:
                self.initialize_cell_measurement(measurement)
        if gene_attributes_dict:
            for attribute_name, attribute_value in gene_attributes_dict.items():
                self.initialize_gene_attribute(attribute_name, attribute_value)

        if remap_attributes:
            self.remap_categorical_attributes()

    

    def __len__(self):
        return self.nb_cells

    def __getitem__(self, idx):
        """Implements @abstractcmethod in ``torch.utils.data.dataset.Dataset`` ."""
        return idx

    

    @X.setter
    def X(self, X: Union[np.ndarray, sp_sparse.csr_matrix]):
        """Sets the data attribute ``X`` and (re)computes the library size."""
        n_dim = len(X.shape)
        if n_dim != 2:
            raise ValueError(
                "Gene expression data should be 2-dimensional not {}-dimensional.".format(
                    n_dim
                )
            )
        self._X = X
        logger.info("Computing the library size for the new data")
        self.compute_library_size_batch()

    @property
    def nb_cells(self) -> int:
        return self.X.shape[0]

    @property
    def nb_genes(self) -> int:
        return self.X.shape[1]

    @property
    def batch_indices(self) -> np.ndarray:
        return self._batch_indices

    @batch_indices.setter
    def batch_indices(self, batch_indices: Union[List[int], np.ndarray]):
        """Sets batch indices and the number of batches."""
        batch_indices = np.asarray(batch_indices, dtype=np.uint16).reshape(-1, 1)
        self.n_batches = len(np.unique(batch_indices))
        self._batch_indices = batch_indices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, labels: Union[List[int], np.ndarray]):
        """Sets labels and the number of labels"""
        labels = np.asarray(labels, dtype=np.uint16).reshape(-1, 1)
        self.n_labels = len(np.unique(labels))
        self._labels = labels


    def remap_categorical_attributes(
        self, attributes_to_remap: Optional[List[str]] = None
    ):
        if attributes_to_remap is None:
            attributes_to_remap = self.cell_categorical_attribute_names

        for attribute_name in attributes_to_remap:
            logger.info("Remapping %s to [0,N]" % attribute_name)
            attr = getattr(self, attribute_name)
            mappings_dict = {
                name: getattr(self, name)
                for name in self.attribute_mappings[attribute_name]
            }
            new_attr, _, new_mappings_dict = remap_categories(
                attr, mappings_dict=mappings_dict
            )
            setattr(self, attribute_name, new_attr)
            for name, mapping in new_mappings_dict.items():
                setattr(self, name, mapping)

    '''def register_dataset_version(self, version_name):
        """Registers a version of the dataset, e.g normalized version."""
        self.dataset_versions.add(version_name)'''

    def initialize_cell_attribute(self, attribute_name, attribute, categorical=False):
        """Sets and registers a cell-wise attribute, e.g annotation information."""
        if attribute_name in self.protected_attributes:
            valid_attribute_name = attribute_name + "_cell"
            logger.warning(
                "{} is a protected attribute and cannot be set with this name "
                "in initialize_cell_attribute, changing name to {} and setting".format(
                    attribute_name, valid_attribute_name
                )
            )
            attribute_name = valid_attribute_name
        try:
            len_attribute = attribute.shape[0]
        except AttributeError:
            len_attribute = len(attribute)
        if not self.nb_cells == len_attribute:
            raise ValueError(
                "Number of cells ({n_cells}) and length of cell attribute ({n_attr}) mismatch".format(
                    n_cells=self.nb_cells, n_attr=len_attribute
                )
            )
        setattr(
            self,
            attribute_name,
            np.asarray(attribute)
            if not isinstance(attribute, sp_sparse.csr_matrix)
            else attribute,
        )
        self.cell_attribute_names.add(attribute_name)
        if categorical:
            self.cell_categorical_attribute_names.add(attribute_name)

    def initialize_gene_attribute(self, attribute_name, attribute):
        """Sets and registers a gene-wise attribute, e.g annotation information."""
        if attribute_name in self.protected_attributes:
            valid_attribute_name = attribute_name + "_gene"
            logger.warning(
                "{} is a protected attribute and cannot be set with this name "
                "in initialize_cell_attribute, changing name to {} and setting".format(
                    attribute_name, valid_attribute_name
                )
            )
        if not self.nb_genes == len(attribute):
            raise ValueError(
                "Number of genes ({n_genes}) and length of gene attribute ({n_attr}) mismatch".format(
                    n_genes=self.nb_genes, n_attr=len(attribute)
                )
            )
        setattr(self, attribute_name, attribute)
        self.gene_attribute_names.add(attribute_name)

    def initialize_mapped_attribute(
        self, source_attribute_name, mapping_name, mapping_values
    ):
        """Sets and registers an attribute mapping, e.g labels to named cell_types."""
        source_attribute = getattr(self, source_attribute_name)

        if isinstance(source_attribute, np.ndarray):
            type_source = source_attribute.dtype
        else:
            element = source_attribute[0]
            while isinstance(element, list):
                element = element[0]
            type_source = type(source_attribute[0])
        if not np.issubdtype(type_source, np.integer):
            raise ValueError(
                "Mapped attribute {attr_name} should be categorical not {type}".format(
                    attr_name=source_attribute_name, type=type_source
                )
            )
        cat_max = np.max(source_attribute)
        if not cat_max <= len(mapping_values):
            raise ValueError(
                "Max value for {attr_name} ({cat_max}) is higher than {map_name} ({n_map}) mismatch".format(
                    attr_name=source_attribute_name,
                    cat_max=cat_max,
                    map_name=mapping_name,
                    n_map=len(mapping_values),
                )
            )
        self.attribute_mappings[source_attribute_name].append(mapping_name)
        setattr(self, mapping_name, mapping_values)

    def compute_library_size_batch(self):
        self.local_means = np.zeros((self.nb_cells, 1))
        self.local_vars = np.zeros((self.nb_cells, 1))
        for i_batch in range(self.n_batches):
            idx_batch = np.squeeze(self.batch_indices == i_batch)
            self.local_means[idx_batch], self.local_vars[
                idx_batch
            ] = compute_library_size(self.X[idx_batch])
            #print('compute', i_batch, idx_batch.shape, self.X[idx_batch].shape, self.batch_indices)
        self.cell_attribute_names.update(["local_means", "local_vars"])

    def collate_fn_builder(
        self,
        add_attributes_and_types: Dict[str, type] = None,
        override: bool = False,
        corrupted=False,
    ) -> Callable[[Union[List[int], np.ndarray]], Tuple[torch.Tensor, ...]]:
        """Returns a collate_fn with the requested shape/attributes"""

        if override:
            attributes_and_types = dict()
        else:
            attributes_and_types = dict(
                [
                    ("X", np.float32) if not corrupted else ("corrupted_X", np.float32),
                    ("local_means", np.float32),
                    ("local_vars", np.float32),
                    ("batch_indices", np.int64),
                    ("labels", np.int64),
                ]
            )

        if add_attributes_and_types is None:
            add_attributes_and_types = dict()
        attributes_and_types.update(add_attributes_and_types)
        return partial(self.collate_fn_base, attributes_and_types)

    def collate_fn_base(
        self, attributes_and_types: Dict[str, type], batch: Union[List[int], np.ndarray]
    ) -> Tuple[torch.Tensor, ...]:
        """Given indices and attributes to batch, returns a full batch of ``Torch.Tensor``"""
        indices = np.asarray(batch)
        data_numpy = [
            getattr(self, attr)[indices].astype(dtype)
            if isinstance(getattr(self, attr), np.ndarray)
            else getattr(self, attr)[indices].toarray().astype(dtype)
            for attr, dtype in attributes_and_types.items()
        ]

        data_torch = tuple(torch.from_numpy(d) for d in data_numpy)
        return data_torch


def remap_categories(
    original_categories: Union[List[int], np.ndarray],
    mapping_from: Union[List[int], np.ndarray] = None,
    mapping_to: Union[List[int], np.ndarray] = None,
    mappings_dict: Dict[str, Union[List[str], List[int], np.ndarray]] = None,
) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Dict[str, np.ndarray]]]:
    original_categories = np.asarray(original_categories)
    unique_categories = list(np.unique(original_categories))
    n_categories = len(unique_categories)
    if mapping_to is None:
        mapping_to = list(range(n_categories))
    if mapping_from is None:
        mapping_from = unique_categories

    # check lengths
    if n_categories > len(mapping_from):
        raise ValueError(
            "Size of provided mapping_from greater than the number of categories."
        )
    if len(mapping_to) != len(mapping_from):
        raise ValueError("Length mismatch between mapping_to and mapping_from.")

    new_categories = np.copy(original_categories)
    for cat_from, cat_to in zip(mapping_from, mapping_to):
        new_categories[original_categories == cat_from] = cat_to
    new_categories = new_categories.astype(np.uint16)
    unique_new_categories = np.unique(new_categories)
    if mappings_dict is not None:
        new_mappings = {}
        for mapping_name, mapping in mappings_dict.items():
            new_mapping = np.empty(
                unique_new_categories.shape[0], dtype=np.asarray(mapping).dtype
            )
            for cat_from, cat_to in zip(mapping_from, mapping_to):
                new_mapping[cat_to] = mapping[cat_from]
            new_mappings[mapping_name] = new_mapping
        return new_categories, n_categories, new_mappings
    else:
        return new_categories, n_categories


def compute_library_size(
    data: Union[sp_sparse.csr_matrix, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    if np.ma.is_masked(masked_log_sum):
        logger.warning(
            "This dataset has some empty cells, this might fail scVI inference."
            "Data should be filtered with `my_dataset.filter_cells_by_count()"
        )
    log_counts = masked_log_sum.filled(0)
    local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
    local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
    #print('compute library', data.shape, log_counts, sum_counts, local_var)
    return local_mean, local_var

    
class SCDataset(GeneExpressionDataset):

    def __init__(
        self,
        filename: str,
        save_path: str = "data/",
        url: str = None,
        new_n_genes: int = None,
        subset_genes: Iterable[Union[int, str]] = None,
        compression: str = None,
        sep: str = ",",
        gene_by_cell: bool = True,
        labels_file: str = None,
        batch_ids_file: str = None,
        delayed_populating: bool = False,
        mat = None,
        ylabels = None,
        tlabels = None,
        cell_types = None,
        batch_ids = None
    ):
        self.compression = compression
        self.sep = sep
        self.gene_by_cell = (
            gene_by_cell
        )  # Whether the original dataset is genes by cells
        self.labels_file = labels_file
        self.batch_ids_file = batch_ids_file
        self.mat = mat
        self.ylabels = ylabels
        self.tlabels = tlabels
        self.cell_types = cell_types
        self.batch_ids = batch_ids
        
        super().__init__()
        '''super().__init__(
            urls=url,
            filenames=filename,
            save_path=save_path,
            delayed_populating=delayed_populating,
        )'''
        #self.subsample_genes(new_n_genes, subset_genes)
        self.populate()

    def populate(self):
        #gene_names = np.asarray(data.columns, dtype=str)
        gene_names = [str(i) for i in range(self.mat.shape[1])]
        batch_indices = None

        if self.batch_ids_file is not None:
            batch_indices = pd.read_csv(
                os.path.join(self.save_path, self.batch_ids_file), header=0, delimiter='\t'
            )[['barcode', 'info']]
            batch_indices['info'] = batch_indices['info'].apply(lambda x: int(x[-1])-1)
            batch_indices = batch_indices.values
            
        if self.batch_ids is not None:
            batch_indices = np.array(self.batch_ids).reshape(-1, 1)
            #print('batch indices', batch_indices.shape)

        #print('labels', self.ylabels, self.cell_types, batch_indices, gene_names)
        self.populate_from_data(
            X=self.mat,
            batch_indices=batch_indices,
            labels=None,
            gene_names=gene_names,
            cell_types=self.cell_types,
        )
        #self.filter_cells_by_count()

