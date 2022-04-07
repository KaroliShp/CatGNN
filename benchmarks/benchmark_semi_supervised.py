from benchmarks.utils.train_semi_supervised import train_eval_loop
import torch
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric
from benchmarks.models.semi_supervised_models.gcn_models import GCN_1, Factored_GCN_1, GCN_2, Factored_GCN_2, GCN_2_Forwards, PyG_GCN, PyG_GCN_Paper, GCN_2_Paper
import numpy as np
from benchmarks.utils.analyse_performance import analyse_repeated_benchmark


def run_benchmark(dataset_name, model_nn, split='public', normalize=True,
                  lr=0.01, weight_decay=5e-4, num_epochs=100, debug=True, 
                  sender_to_receiver=True, **kwargs):
    dataset = PlanetoidDataset(dataset_name, split, normalize)

    train_y, val_y, test_y = dataset.split()
    train_mask, val_mask, test_mask = dataset.get_split_masks()

    input_dim, output_dim = dataset.get_dimensions()

    V = dataset.get_vertices()
    E = dataset.get_edges(sender_to_receiver)
    X = dataset.get_features()

    model = model_nn(input_dim=input_dim, output_dim=output_dim, **kwargs)

    return train_eval_loop(model, V, E, X, train_y, train_mask, 
                           val_y, val_mask, test_y, test_mask, 
                           lr, weight_decay, num_epochs, debug)


def repeat_benchmark(repeat, func, *args, **kwargs):
    train_accs = []
    val_accs = []
    test_accs = []
    avg_runtimes = []

    for r in range(repeat):
        print(f'r: {r}')
        training_stats, avg_runtime = func(*args, **kwargs)
        train_accs.append(training_stats['train_acc'][-1])
        val_accs.append(training_stats['val_acc'][-1])
        test_accs.append(training_stats['test_acc'][-1])
        avg_runtimes.append(avg_runtime)

    return analyse_repeated_benchmark(train_accs, val_accs, test_accs, avg_runtimes, repeat)


def run_all_benchmarks(repeat=5):
    # Cora benchmarks for a single layer
    experiment_1 = 'Cora GCN 1 Layer'
    train_res, val_res, test_res, runtime_res = repeat_benchmark(repeat, run_benchmark, 'Cora', GCN_2_Paper)
    
    print(f'avg_train_acc: {train_res[0]}, std: {train_res[1]}, sem: {train_res[2]}')
    print(f'avg_val_acc: {val_res[0]}, std: {val_res[1]}, sem: {val_res[2]}')
    print(f'avg_test_acc: {test_res[0]}, std: {test_res[1]}, sem: {test_res[2]}')
    print(f'avg_runtime: {runtime_res[0]}, std: {runtime_res[1]}, sem: {runtime_res[2]}')

    #experiment_2 = 'Cora PyGCN 1 Layer'
    #avg_train_acc_2, avg_val_acc_2, avg_test_acc_2, avg_runtime_2 = repeat_benchmark(repeat, run_benchmark, 'Cora', GCN_2)




    # CiteSeer benchmarks for a single layer
    # TODO

    # PubMed benchmarks for a single layer
    # TODO

    # Cora benchmarks for how it scales with several layers
    # TODO


if __name__ == '__main__':
    run_all_benchmarks()