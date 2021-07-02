from .base_split_manager import BaseSplitManager
from .class_disjoint_split_manager import ClassDisjointSplitManager
from .index_split_manager import IndexSplitManager

from torchvision import transforms
import torch
import numpy as np
import torch

class CustomSplitManager(BaseSplitManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_names = ["test"]

    def _create_split_schemes(self, datasets):
        output = {}
        for transform_type, v1 in datasets.items():
            output[transform_type] = {}
            for split_name, v2 in v1.items():
                indices = v2.get_split_indices(split_name)
#                 print(split_name)
                if indices is not None:
                    output[transform_type][split_name] = torch.utils.data.Subset(v2, indices)
                else:
                    output[transform_type][split_name] = v2
        return {self.get_split_scheme_name(0): output}

    def get_test_set_name(self):
        return 'UsingOriginalTest'

    def get_base_split_scheme_name(self):
        return self.get_test_set_name()

    def get_split_scheme_name(self, partition):
        return self.get_base_split_scheme_name()

    def split_assertions(self):
        pass
    
    
