"""
ogbn-arxiv dataset from OGB:
https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv

This benchmark code from:
https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
"""

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator # Sometimes buggy? Something wrong with acquiring lock in OGB

import argparse
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv

from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2, GCNLayer_Factored_MPNN_2, GCNLayer_MPNN_2_Forwards
from catgnn.layers.gat.gat_mpnn_2 import GATLayer_MPNN_2
from catgnn.layers.sage.sage_mpnn_2 import SAGELayer_MPNN_2
from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2


device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


"""
GCN
"""


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, factored=False, forwards=False):
        super(GCN, self).__init__()

        if factored:
            print('Factored GCN')
            layer = GCNLayer_Factored_MPNN_2
        elif forwards:
            print('Forwards GCN')
            layer = GCNLayer_MPNN_2_Forwards
        else:
            print('Backwards GCN')
            layer = GCNLayer_MPNN_2

        self.convs = torch.nn.ModuleList()
        self.convs.append(layer(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                layer(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(layer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        V = torch.arange(0, x.shape[0], dtype=torch.int64).to(device) # Create vertices
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(V, adj_t, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](V, adj_t, x)
        return x.log_softmax(dim=-1)


class PyG_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PyG_GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


"""
GraphSAGE
"""


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGELayer_MPNN_2(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGELayer_MPNN_2(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGELayer_MPNN_2(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        V = torch.arange(0, x.shape[0], dtype=torch.int64) # Create vertices
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(V, adj_t, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](V, adj_t, x)
        return x.log_softmax(dim=-1)


class PyG_SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PyG_SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


"""
GAT
"""


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATLayer_MPNN_2(in_channels, hidden_channels, heads=2))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATLayer_MPNN_2(hidden_channels*2, hidden_channels, heads=2))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATLayer_MPNN_2(hidden_channels*2, out_channels, heads=1))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        V = torch.arange(0, x.shape[0], dtype=torch.int64).to(device) # Create vertices
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(V, adj_t, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](V, adj_t, x)
        return x.log_softmax(dim=-1)


class PyG_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PyG_GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=2))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*2, hidden_channels, heads=2))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels*2, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


"""
SGC
"""


class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGCLayer_MPNN_2(in_channels, hidden_channels, K=2))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGCLayer_MPNN_2(hidden_channels, hidden_channels, K=2))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SGCLayer_MPNN_2(hidden_channels, out_channels, K=2))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        V = torch.arange(0, x.shape[0], dtype=torch.int64).to(device) # Create vertices
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(V, adj_t, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](V, adj_t, x)
        return x.log_softmax(dim=-1)


class PyG_SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PyG_GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConv(in_channels, hidden_channels, K=2, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels, K=2, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SGConv(hidden_channels, out_channels, K=2, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


"""
Logger imported from
https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/logger.py
"""

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


def train(model, data, train_idx, optimizer, is_sparse):
    model.train()

    optimizer.zero_grad()
    if is_sparse:
        out = model(data.x, data.adj_t)[train_idx]
    else:
        out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, is_sparse):
    model.eval()

    if is_sparse:
        out = model(data.x, data.adj_t)
    else:
        out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
        parser.add_argument('--device', type=int, default=0)
        parser.add_argument('--log_steps', type=int, default=1)
        parser.add_argument('--use_sage', type=bool, default=False)
        parser.add_argument('--use_gat', type=bool, default=False)
        parser.add_argument('--use_sgc', type=bool, default=False)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--hidden_channels', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=500)
        parser.add_argument('--runs', type=int, default=10)
        parser.add_argument('--catgnn', type=bool, default=False)
        parser.add_argument('--catgnn_factored', type=bool, default=False)
        parser.add_argument('--catgnn_forward', type=bool, default=False)
        args = parser.parse_args(args=[])
    print(args)
    
    # Prepare data (differently from the original benchmark - don't use pytorch_sparse for CatGNN)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                    transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    if args.catgnn:
        # Turn SparseTensor back to normal Tensor - I think this way to convert may cause some issues with memory
        # TODO: take a closer look into this
        print('Convert SparseTensor')
        data.edge_index = torch.cat((data.adj_t.coo()[0],data.adj_t.coo()[1])).view(2,-1)
        data.adj_t = None
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        if args.catgnn:
            print('--CATGNN SAGE--')
            model = SAGE(data.num_features, args.hidden_channels,
                         dataset.num_classes, args.num_layers, 
                         args.dropout).to(device)
        else:
            print('--PyG SAGE--')
            model = PyG_SAGE(data.num_features, args.hidden_channels,
                             dataset.num_classes, args.num_layers,
                             args.dropout).to(device)
    elif args.use_gat:
        if args.catgnn:
            print('--CATGNN GAT--')
            model = GAT(data.num_features, args.hidden_channels,
                        dataset.num_classes, args.num_layers, 
                        args.dropout).to(device)
        else:
            print('--PyG GAT--')
            model = PyG_GAT(data.num_features, args.hidden_channels,
                             dataset.num_classes, args.num_layers,
                             args.dropout).to(device)
    elif args.use_sgc:
        if args.catgnn:
            print('--CATGNN SGC--')
            model = SGC(data.num_features, args.hidden_channels,
                        dataset.num_classes, args.num_layers, 
                        args.dropout).to(device)
        else:
            print('--PyG SGC--')
            model = PyG_SGC(data.num_features, args.hidden_channels,
                            dataset.num_classes, args.num_layers,
                            args.dropout).to(device)
    else:
        if args.catgnn:
            print('--CATGNN GCN--')
            model = GCN(data.num_features, args.hidden_channels,
                        dataset.num_classes, args.num_layers,
                        args.dropout, args.catgnn_factored, 
                        args.catgnn_forward).to(device)
        else:
            print('--PyG GCN--')
            model = PyG_GCN(data.num_features, args.hidden_channels,
                            dataset.num_classes, args.num_layers,
                            args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            start = timer()
            loss = train(model, data, train_idx, optimizer, not args.catgnn)
            end = timer()

            result = test(model, data, split_idx, evaluator, not args.catgnn)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}% '
                      f'Runtime: {timedelta(seconds=end - start)}')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()