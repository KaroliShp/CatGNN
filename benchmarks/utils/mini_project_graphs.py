import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


def profiling_graphs():
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))

    gcn_profile = [0.000024, 0.000202, 0.017416, 0.047481]
    gcn_labels = ['pullback', 'kernel', 'pushforward', 'aggregator']

    gat_profile = [0.000076, 0.401932, 0.019377, 0.064818, 0.000067]
    gat_labels = ['pullback', 'k\'', 'pushforward', 'aggregator', '(id,*)']

    ax1.pie(gcn_profile, explode=[0,0,0.05,0.05])
    ax1.axis('equal')
    ax1.legend(gcn_labels, fontsize=14)
    ax1.set_title('GCN runtime profile', fontsize=30)

    ax2.pie(gat_profile, explode=[0,0,0.05,0.05,0])
    ax2.axis('equal')
    ax2.legend(gat_labels, fontsize=14)
    ax2.set_title('GAT runtime profile', fontsize=30)

    fig.tight_layout()
    fig.savefig("profiling_graphs.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    profiling_graphs()