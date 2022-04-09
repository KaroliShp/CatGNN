import torch
from torch import nn
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np  # for statistics


def train(V, E, X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(V, E, X)[mask]
    loss = nn.functional.cross_entropy(y_hat, y) # TODO - PyG uses nll here
    loss.backward()
    optimiser.step()
    return loss.data


def evaluate(V, E, X, y, mask, model):
    model.eval()
    y_hat = model(V, E, X)[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct/num_total)
    return accuracy.item()


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


def train_eval_loop(model, V, E, X, train_y, train_mask, 
                    valid_y, valid_mask, test_y, test_mask,
                    lr: float, weight_decay: float, num_epochs: int, debug: bool):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_stats = None
    runtimes = []

    for epoch in range(num_epochs):
        start = timer()
        train_loss = train(V, E, X, train_y, train_mask, model, optimiser)
        end = timer()
        runtimes.append(timedelta(seconds=end-start))

        train_acc = evaluate(V, E, X, train_y, train_mask, model)
        valid_acc = evaluate(V, E, X, valid_y, valid_mask, model)
        test_acc = evaluate(V, E, X, test_y, test_mask, model)

        if (epoch % 10 == 0 or epoch == num_epochs-1) and debug:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}, test accuracy: {test_acc:.3f}, time: {runtimes[-1]}")
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'test_acc': test_acc, 'epoch': epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    
    avg_runtime = np.mean(runtimes)
    return training_stats, avg_runtime