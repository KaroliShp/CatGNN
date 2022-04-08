from benchmarks.utils.train_semi_supervised import train_eval_loop
from catgnn.datasets.planetoid import PlanetoidDataset
from catgnn.datasets.pyg_supervised_dataset import get_TU_dataset
from benchmarks.utils.pyg_train_supervised import cross_validation_with_val_set
from benchmarks.models.supervised.gcn_models import GCN_2, PyG_GCN
from benchmarks.models.supervised.gin_models import GIN_2, PyG_GIN, GIN0_2, PyG_GIN0
from benchmarks.models.supervised.sage_models import GraphSAGE_2, PyG_GraphSAGE
from benchmarks.utils.analyse_performance import analyse_repeated_benchmark, stringify_statistics


def run_benchmark(name, model_nn, num_layers, num_hidden_units, 
                  folds=10, batch_size=128, lr=0.01, weight_decay=0, 
                  lr_decay_factor=0.5, lr_decay_step_size=50, num_epochs=100, 
                  debug=True, **kwargs):
    dataset = get_TU_dataset(name)

    model = model_nn(dataset, num_layers, num_hidden_units, **kwargs)

    return cross_validation_with_val_set(
        dataset,
        model,
        folds=folds,
        epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay_factor=lr_decay_factor,
        lr_decay_step_size=lr_decay_step_size,
        weight_decay=weight_decay,
        debug=debug
    )


def run_paper_benchmarks(name, num_layers, num_hidden_units):
    """
    run_benchmark(name, GIN0_2, num_layers, num_hidden_units)
    run_benchmark(name, PyG_GIN0, num_layers, num_hidden_units)
    """
    """
    run_benchmark(name, GIN_2, num_layers, num_hidden_units)
    run_benchmark(name, PyG_GIN, num_layers, num_hidden_units)
    """
    """
    run_benchmark(name, GCN_2, num_layers, num_hidden_units)
    run_benchmark(name, PyG_GCN, num_layers, num_hidden_units)
    """
    run_benchmark(name, GraphSAGE_2, num_layers, num_hidden_units)
    #run_benchmark(name, PyG_GraphSAGE, num_layers, num_hidden_units)
    


if __name__ == '__main__':
    run_paper_benchmarks('REDDIT-BINARY', 1, 16)