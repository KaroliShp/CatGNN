import torch
from timeit import default_timer as timer
from datetime import timedelta


def train(loader, model, optimiser):
    model.train()

    total_loss = 0
    for data in loader:
        optimiser.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.functional.nll_loss(output, data.y)
        loss.backward()
        optimiser.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader, model):
    model.eval()

    total_correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
    return total_correct / len(loader.dataset)


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


# Training loop
def train_eval_loop(model, train_loader, test_loader):
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    training_stats = None

    # Training loop
    for epoch in range(100):
        start = timer()
        train_loss = train(train_loader, model, optimiser)
        end = timer()
        print(f'Time: {timedelta(seconds=end-start)}')

        train_acc = evaluate(train_loader, model)
        test_acc = evaluate(test_loader, model)

        if epoch % 1 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} test accuracy: {test_acc:.3f}")
        # Store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'test_acc': test_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    
    # Lets look at our final test performance
    test_acc = evaluate(test_loader, model)
    print(f"Final test accuracy: {test_acc:.3f}")
    return training_stats