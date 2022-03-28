import torch
from torch import nn
from timeit import default_timer as timer
from datetime import timedelta


def train_gnn_cora(V_data, E_data, X_data, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(V_data, E_data, X_data)[mask]
    loss = nn.functional.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data


def evaluate_gnn_cora(V_data, E_data, X_data, y, mask, model):
    model.eval()
    y_hat = model(V_data, E_data, X_data)[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct/num_total)
    return accuracy


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


# Training loop
def train_eval_loop(model, V_data, E_data, X_data, train_y, train_mask, 
                    V_val, E_val, X_val, valid_y, valid_mask, 
                    V_test, E_test, X_test, test_y, test_mask):
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    training_stats = None
    # Training loop
    for epoch in range(100):
        start = timer()
        train_loss = train_gnn_cora(V_data, E_data, X_data, train_y, train_mask, model, optimiser)
        end = timer()
        print(f'Time: {timedelta(seconds=end-start)}')
        train_acc = evaluate_gnn_cora(V_data, E_data, X_data, train_y, train_mask, model)
        valid_acc = evaluate_gnn_cora(V_val, E_val, X_val, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc = evaluate_gnn_cora(V_test, E_test, X_test, test_y, test_mask, model)
    print(f"Our final test accuracy for the SimpleGNN is: {test_acc:.3f}")
    return training_stats