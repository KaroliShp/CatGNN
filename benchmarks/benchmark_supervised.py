from benchmarks.dataset import get_dataset
from benchmarks.train_test import cross_validation_with_val_set
from benchmarks.models.supervised.gcn_models import GCN_2, PyG_GCN
from benchmarks.models.supervised.gin_models import GIN_2, PyG_GIN, GIN0_2, PyG_GIN0


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


results = []
def run_test(dataset_name, Net, num_layers, hidden):
    print(f'--\n{dataset_name} - {Net.__name__}')
    
    dataset = get_dataset(dataset_name, sparse=True)
    model = Net(dataset, num_layers, hidden)
    loss, acc, std = cross_validation_with_val_set(
        dataset,
        model,
        folds=10,
        epochs=100,
        batch_size=128,
        lr=0.01,
        lr_decay_factor=0.5,
        lr_decay_step_size=50,
        weight_decay=0,
        logger=None,
    )
    print(f'\n--Loss: {loss}, acc: {acc}, std: {std}--\n')

if __name__ == '__main__':
    layers = [1, 2, 3, 4, 5]
    hiddens = [16, 32, 64, 128]

    #run_test('MUTAG', GCN_2, 1, 16)
    #run_test('MUTAG', PyG_GCN, 1, 16)
    run_test('MUTAG', GIN_2, 1, 16)
    run_test('MUTAG', PyG_GIN, 1, 16)
