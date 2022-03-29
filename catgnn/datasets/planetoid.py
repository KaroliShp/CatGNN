from torch_geometric.datasets import Planetoid
import torch
from catgnn.datasets.dataset import Dataset

class PlanetoidDataset(Dataset):

    def __init__(self, name):
        super(PlanetoidDataset, self).__init__()

        assert name in ["Cora", "CiteSeer", "PubMed"]
        cora_pyg = Planetoid(root=f'/tmp/{name}', name=name, split="full")
        self.cora_data = cora_pyg[0]

    def train_val_test_split(self):
        train_x = self.cora_data.x[self.cora_data.train_mask]
        train_y = self.cora_data.y[self.cora_data.train_mask]

        valid_x = self.cora_data.x[self.cora_data.val_mask]
        valid_y = self.cora_data.y[self.cora_data.val_mask]

        test_x = self.cora_data.x[self.cora_data.test_mask]
        test_y = self.cora_data.y[self.cora_data.test_mask]

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_split_masks(self):
        return self.cora_data.train_mask, self.cora_data.val_mask, self.cora_data.test_mask

    def get_features(self):
        return self.cora_data.x
    
    def get_edges(self, sender_to_receiver=True):
        if sender_to_receiver:
            return self.cora_data.edge_index
        else:
            E_swapped = self.cora_data.edge_index.T
            E_swapped = E_swapped[E_swapped[:, 1].sort()[1]]
            return torch.cat((E_swapped[:,1].view(-1,1), E_swapped[:,0].view(-1,1)), dim=1).T    
    
    def get_vertices(self):
        V_data = set()
        for e in self.cora_data.edge_index.T.numpy():
            V_data.add(e[0])
        V_data = torch.tensor(list(V_data), dtype=torch.int64)
        return V_data

    def validate_dataset(self):
        raise NotImplementedError