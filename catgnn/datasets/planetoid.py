import torch
from catgnn.datasets.semi_supervised_dataset import SemiSupervisedDataset
import torch_geometric

class PlanetoidDataset(SemiSupervisedDataset):

    def __init__(self, name, split, normalize=False):
        super(PlanetoidDataset, self).__init__()

        assert name in ["Cora", "CiteSeer", "PubMed"]
        assert split in ["full", "public", "random"]

        if normalize:
            self.dataset_obj = torch_geometric.datasets.Planetoid(root=f'/tmp/{name}', name=name, split=split, transform=torch_geometric.transforms.NormalizeFeatures())
        else:
            self.dataset_obj = torch_geometric.datasets.Planetoid(root=f'/tmp/{name}', name=name, split=split)
        self.dataset = self.dataset_obj[0]

    def split(self):
        train_y = self.dataset.y[self.dataset.train_mask]
        valid_y = self.dataset.y[self.dataset.val_mask]
        test_y = self.dataset.y[self.dataset.test_mask]
        return train_y, valid_y, test_y

    def get_split_masks(self):
        return self.dataset.train_mask, self.dataset.val_mask, self.dataset.test_mask

    def get_features(self):
        return self.dataset.x
    
    def get_edges(self, sender_to_receiver=True):
        if sender_to_receiver: # TODO
            return self.dataset.edge_index
        else:
            E_swapped = self.dataset.edge_index.T
            E_swapped = E_swapped[E_swapped[:, 1].sort()[1]]
            return torch.cat((E_swapped[:,1].view(-1,1), E_swapped[:,0].view(-1,1)), dim=1).T    
    
    def get_vertices(self):
        return torch.arange(0, self.dataset.x.shape[0], dtype=torch.int64)

    def get_dimensions(self):
        return self.dataset_obj.num_features, self.dataset_obj.num_classes