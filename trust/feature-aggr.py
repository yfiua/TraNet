from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr

def graph_to_sparse_matrix(G):
    n = G.vcount()
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    if not G.is_directed():
        xs, ys = np.hstack((xs, ys)).T, np.hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T
    return coo_matrix((np.ones(xs.shape), (xs, ys)), shape=(n, n), dtype=np.int16)

def read_graph(dataset):
    # dataset (network)
    df = pd.read_csv('data/' + dataset, sep='\t|,', header=None, engine='python')

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    num_nodes = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)

    return G

# main
def main():
    # params
    datasets = [ 'de-wiki-talk', 'fr-wiki-talk', 'boards.ie-network', 'sag-network' ]
    features = [ 'degree' , 'clustering_coefficient', 'pagerank' ] #, 'eccentricity',  'strength' , 'diversity' ]  'personalized_pagerank',
    R = 10

    # result matrix
    res = np.zeros((len(datasets), len(features), 10))

    for i in range(len(datasets)):
        G = read_graph(datasets[i])

        # feature matrix
        F = [ G.degree(), G.transitivity_local_undirected(mode='zero'), G.pagerank() ]

        # adjacency matrix (simplified)
        A = graph_to_sparse_matrix(G.as_undirected().simplify())
        d = np.squeeze(np.array(A.sum(axis=1)))
        d[d == 0] = 1

        for f in range(len(features)):
            v = F[f]
            v = v / np.linalg.norm(v)

            v_old = np.array([v])
            for r in range(R):
                v = A.dot(v) / d
                v = v / np.linalg.norm(v)
                #print max([ abs(pearsonr(v, u)[0]) for u in v_old])
                res[i, f, r] = max([ abs(pearsonr(v, u)[0]) for u in v_old])
                v_old = np.vstack((v_old, v))

    np.save('res/eval-feature-aggr.npy', res)

if __name__ == '__main__':
    # init
    main()

