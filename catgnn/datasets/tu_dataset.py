from torch_geometric.datasets import TUDataset
import torch
from catgnn.datasets.dataset import Dataset

class TUDatasetDataset(Dataset):

    def __init__(self, name='MUTAG'):
        super(TUDatasetDataset, self).__init__()

        assert name in ['MUTAG', 'ENZYMES', 'PROTEINS']
        dataset = TUDataset(root=f'/tmp/{name}', name=name).shuffle()
        self.dataset = dataset[0]

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


if __name__ == '__main__':
    dataset = TUDatasetDataset()