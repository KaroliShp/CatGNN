import torch

class Dataset:

    def __init__(self):
        super(Dataset, self).__init__()

    def train_val_test_split(self):
        raise NotImplementedError

    def get_split_masks(self):
        raise NotImplementedError

    def get_features(self):
        raise NotImplementedError
    
    def get_edges(self, sender_to_receiver=True):
        raise NotImplementedError
    
    def get_vertices(self):
        raise NotImplementedError

    def validate_dataset(self):
        raise NotImplementedError