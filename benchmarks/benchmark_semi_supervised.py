from benchmarks.utils.train_semi_supervised import train_eval_loop
import torch
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric
from benchmarks.models.semi_supervised_models.gcn_models import (
    GCN_1,
    GCN_2,
    PyG_GCN,
    GCN_1_Paper,
    GCN_2_Paper,
    PyG_GCN_Paper,
)
from benchmarks.models.semi_supervised_models.sgc_models import (
    SGC_2_Paper,
    PyG_SGC_Paper,
)
from benchmarks.models.semi_supervised_models.gat_models import (
    GAT_2_Paper,
    PyG_GAT_Paper,
)
import numpy as np
from benchmarks.utils.analyse_performance import (
    analyse_repeated_benchmark,
    stringify_statistics,
)


device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def run_benchmark(
    dataset_name,
    model_nn,
    split="public",
    normalize=True,
    lr=0.01,
    weight_decay=5e-4,
    num_epochs=200,
    debug=False,
    sender_to_receiver=True,
    **kwargs,
):
    dataset = PlanetoidDataset(dataset_name, split, normalize)

    train_y, val_y, test_y = dataset.split()
    train_mask, val_mask, test_mask = dataset.get_split_masks()

    input_dim, output_dim = dataset.get_dimensions()

    V = dataset.get_vertices()
    E = dataset.get_edges(sender_to_receiver)
    X = dataset.get_features()

    model = model_nn(input_dim=input_dim, output_dim=output_dim, **kwargs).to(device)

    return train_eval_loop(
        model,
        V,
        E,
        X,
        train_y,
        train_mask,
        val_y,
        val_mask,
        test_y,
        test_mask,
        lr,
        weight_decay,
        num_epochs,
        debug,
    )


def repeat_benchmark(repeat, func, experiment_name, *args, **kwargs):
    train_accs = []
    val_accs = []
    test_accs = []
    avg_runtimes = []

    for r in range(repeat):
        training_stats, avg_runtime = func(*args, **kwargs)
        train_accs.append(training_stats["train_acc"][-1])
        val_accs.append(training_stats["val_acc"][-1])
        test_accs.append(training_stats["test_acc"][-1])
        avg_runtimes.append(avg_runtime)
        print(
            f"{experiment_name} repetition: {r}, train_acc: {train_accs[-1]}, val_acc: {val_accs[-1]}, test_acc: {test_accs[-1]}, runtime: {avg_runtimes[-1]}"
        )

    return analyse_repeated_benchmark(
        train_accs, val_accs, test_accs, avg_runtimes, repeat
    )


def run_pytorch_benchmarks_gcn(name="Cora", repeat=2):
    # Public splits
    """
    experiment_0 = f'{name} CatGNN GCN (1)'
    train_res_0, val_res_0, test_res_0, runtime_res_0 = repeat_benchmark(repeat, run_benchmark, experiment_0, name, GCN_1_Paper)
    """

    experiment_1 = f'{name} CatGNN GCN (2), public split'
    train_res_1, val_res_1, test_res_1, runtime_res_1 = repeat_benchmark(repeat, run_benchmark, experiment_1, name, GCN_2_Paper)

    experiment_2 = f'{name} CatGNN GCN (2, factored), public split'
    train_res_2, val_res_2, test_res_2, runtime_res_2 = repeat_benchmark(repeat, run_benchmark, experiment_2, name, GCN_2_Paper, factored=True)

    experiment_3 = f'{name} CatGNN GCN (2, forwards), public split'
    train_res_3, val_res_3, test_res_3, runtime_res_3 = repeat_benchmark(repeat, run_benchmark, experiment_3, name, GCN_2_Paper, forwards=True)

    experiment_4 = f'{name} PyG GCN, public split'
    train_res_4, val_res_4, test_res_4, runtime_res_4 = repeat_benchmark(repeat, run_benchmark, experiment_4, name, PyG_GCN_Paper)

    print('')
    #print(stringify_statistics(experiment_0, train_res_0, val_res_0, test_res_0, runtime_res_0))
    print(stringify_statistics(experiment_1, train_res_1, val_res_1, test_res_1, runtime_res_1))
    print(stringify_statistics(experiment_2, train_res_2, val_res_2, test_res_2, runtime_res_2))
    print(stringify_statistics(experiment_3, train_res_3, val_res_3, test_res_3, runtime_res_3))
    print(stringify_statistics(experiment_4, train_res_4, val_res_4, test_res_4, runtime_res_4))

    """
    # Random splits
    experiment_5 = f'{name} CatGNN GCN (2), random split'
    train_res_5, val_res_5, test_res_5, runtime_res_5 = repeat_benchmark(repeat, run_benchmark, experiment_5, name, GCN_2_Paper, split='random')

    experiment_6 = f'{name} CatGNN GCN (2, factored), random split'
    train_res_6, val_res_6, test_res_6, runtime_res_6 = repeat_benchmark(repeat, run_benchmark, experiment_6, name, GCN_2_Paper, split='random', factored=True)

    experiment_7 = f'{name} CatGNN GCN (2, forwards), random split'
    train_res_7, val_res_7, test_res_7, runtime_res_7 = repeat_benchmark(repeat, run_benchmark, experiment_7, name, GCN_2_Paper, split='random', forwards=True)

    experiment_8 = f'{name} PyG GCN, random split'
    train_res_8, val_res_8, test_res_8, runtime_res_8 = repeat_benchmark(repeat, run_benchmark, experiment_8, name, PyG_GCN_Paper, split='random')

    print('')
    print(stringify_statistics(experiment_5, train_res_5, val_res_5, test_res_5, runtime_res_5))
    print(stringify_statistics(experiment_6, train_res_6, val_res_6, test_res_6, runtime_res_6))
    print(stringify_statistics(experiment_7, train_res_7, val_res_7, test_res_7, runtime_res_7))
    print(stringify_statistics(experiment_8, train_res_8, val_res_8, test_res_8, runtime_res_8))
    """


def run_pytorch_benchmarks_sgc(name="Cora", repeat=2):
    experiment_0 = 'Cora CatGNN SGC (2), public split'
    train_res_0, val_res_0, test_res_0, runtime_res_0 = repeat_benchmark(repeat, run_benchmark, experiment_0, name, SGC_2_Paper, K=3)

    experiment_1 = 'Cora PyG SGC, public split'
    train_res_1, val_res_1, test_res_1, runtime_res_1 = repeat_benchmark(repeat, run_benchmark, experiment_1, name, PyG_SGC_Paper, K=3)

    print('')
    print(stringify_statistics(experiment_0, train_res_0, val_res_0, test_res_0, runtime_res_0))
    print(stringify_statistics(experiment_1, train_res_1, val_res_1, test_res_1, runtime_res_1))


def run_paper_benchmarks_gat(name="Cora", repeat=2):
    experiment_0 = "Cora CatGNN GAT (2), public split"
    train_res_0, val_res_0, test_res_0, runtime_res_0 = repeat_benchmark(
        repeat, run_benchmark, experiment_0, name, GAT_2_Paper
    )

    experiment_1 = "Cora PyG GAT, public split"
    train_res_1, val_res_1, test_res_1, runtime_res_1 = repeat_benchmark(
        repeat, run_benchmark, experiment_1, name, PyG_GAT_Paper
    )

    print("")
    print(
        stringify_statistics(
            experiment_0, train_res_0, val_res_0, test_res_0, runtime_res_0
        )
    )
    print(
        stringify_statistics(
            experiment_1, train_res_1, val_res_1, test_res_1, runtime_res_1
        )
    )


if __name__ == "__main__":
    # run_paper_benchmarks()  # Cora
    # run_paper_benchmarks(name='CiteSeer')
    # run_paper_benchmarks(name='PubMed')
    pass