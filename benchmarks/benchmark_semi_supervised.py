from benchmarks.utils.train_semi_supervised import train_eval_loop
import torch
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric
from benchmarks.models.semi_supervised_models.gcn_models import GCN_1, Factored_GCN_1, GCN_2, Factored_GCN_2, GCN_2_Forwards, PyG_GCN 


def run_benchmark(dataset_name, model_nn, lr=0.01, weight_decay=5e-4, num_epochs=100, debug=True, sender_to_receiver=True):
    dataset = PlanetoidDataset(dataset_name, 'full')

    train_y, val_y, test_y = dataset.split()
    train_mask, val_mask, test_mask = dataset.get_split_masks()

    input_dim, output_dim = dataset.get_dimensions()

    V = dataset.get_vertices()
    E = dataset.get_edges(sender_to_receiver)
    X = dataset.get_features()

    model = model_nn(input_dim=input_dim, output_dim=output_dim)

    return train_eval_loop(model, V, E, X, train_y, train_mask, 
                           val_y, val_mask, test_y, test_mask, 
                           lr, weight_decay, num_epochs, debug)


def run_all_benchmarks(repeat=5):
    # Cora benchmarks for a single layer
    training_stats, avg_runtime = run_benchmark('Cora', GCN_2)
    print(f'Average runtime: {avg_runtime}')
    training_stats, avg_runtime = run_benchmark('Cora', PyG_GCN)
    print(f'Average runtime: {avg_runtime}')

    # CiteSeer benchmarks for a single layer
    # TODO

    # PubMed benchmarks for a single layer
    # TODO

    # Cora benchmarks for how it scales with several layers
    # TODO


if __name__ == '__main__':
    run_all_benchmarks()