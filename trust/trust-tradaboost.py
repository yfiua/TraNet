from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from transform_funcs import *
from tradaboost import *
from utils import *
from scipy.sparse import coo_matrix, linalg

def graph_to_sparse_matrix(G):
    n = G.vcount()
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    if not G.is_directed():
        xs, ys = np.hstack((xs, ys)).T, np.hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T

    try:
        weights = G.es["weights"]
    except KeyError:
        weights = np.ones(xs.shape)

    A = coo_matrix((weights, (xs, ys)), shape=(n, n), dtype=np.int16)

    return A.tocsr()

def get_feature(G, f):
    return _transform_func_degree(getattr(G, f)()) if callable(getattr(G, f)) else _transform_func(getattr(G, f))

# aggregate by the mean value of feature of neighbours
def mean_neighbour(A, d, feature):
    return A.dot(feature) / d

def get_feature_matrix(G, features, rounds=5):
    # local clustering coefficient
    lcc = np.array(G.transitivity_local_undirected(mode='zero'))
    lcc[lcc < 0] = 0                                                 # implementation of igraph is really shitty
    G.clustering_coefficient = lcc

    # compute PageRank
    G_sim = G.copy()
    G_sim = G_sim.simplify(multiple=False)                           # remove loops

    alpha = 0.15
    pagerank = np.array(G_sim.pagerank(damping=1-alpha))
    G.pr = pagerank

    feature_matrix = [ get_feature(G, f) for f in features ]
    X = np.array(feature_matrix).T

    # adjacency matrix (simplified, signs removed)
    A = graph_to_sparse_matrix(G.as_undirected().simplify())
    A = np.abs(A)
    d = np.squeeze(np.array(A.sum(axis=1))).astype(np.int)
    d[d == 0] = 1

    for i in range(rounds):
        feature_matrix = [ mean_neighbour(A, d, f) for f in feature_matrix ]
        X = np.concatenate((X, np.array(feature_matrix).T), axis=1)

    #X = np.hstack((X, np.array([pagerank]).T))
    return X

# read a signed graph in file 'f_graph' and calculate its 'features'
# return its adjacency matrix A, size n, feature matrix X and eigen-trust vector v;
def read_signed_graph(f_graph, features):
    # dataset (graph)
    df = pd.read_csv(f_graph, sep=' ', header=None, skiprows=2)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    n = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    # add signs (+/-)
    G.es["weights"] = df[2].values

    G = G.subgraph(nodes)
    G = G.simplify(multiple=False)                           # remove loops

    # get adjacency matrix
    A = graph_to_sparse_matrix(G)
    v = eigen_trust(A)

    # features
    X = get_feature_matrix(G, features)

    return A, n, np.squeeze(X), v

def read_target_graph(f_graph, features):
    # dataset (graph)
    df = pd.read_csv(f_graph, sep=',', header=None)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    n = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)
    G = G.simplify(multiple=False)                           # remove loops

    # get adjacency matrix
    A = graph_to_sparse_matrix(G)
    v = eigen_trust(A)

    # features
    X = get_feature_matrix(G, features)

    return A, n, np.squeeze(X), v

# main
def main(f_source_graph):
    # params
    f_target_graph = 'sag/sag-network'
    f_target_roles = 'sag/sag-roles'
    f_target_dict = 'sag/sag-dictionary'

    n_trees = 200
    features = [ 'clustering_coefficient' , 'degree' , 'indegree' , 'outdegree', 'pr' ]

    # read datasets
    A_s, n_s, X_s, v_s = read_signed_graph(f_source_graph, features)
    A_t, n_t, X_t, v_t = read_target_graph(f_target_graph, features)

    df_dict = pd.read_csv(f_target_dict, sep=' ')
    df_roles_target = pd.read_csv(f_target_roles, header=None, sep=' ')

    # get users with 'no_captcha' i.e. trusted users
    r = df_roles_target.values
    no_captcha = r[r[:,1]==5][:,0]

    ids = df_dict['ent.string.name'].values     # the user IDs in SAG system
    y_t = np.array([ np.isin(x, no_captcha) for x in ids ]).flatten()

    index_trusted_s = np.argsort(v_s)[::-1][0:500]
    y_s = np.zeros(n_s)
    y_s[index_trusted_s] = 1

    # TraDaBoost
    X_t1, X_t2, y_t1, y_t2 = train_test_split(X_t, y_t, test_size=0.33, random_state=4242)
    v_pred = tradaboost(X_t1, X_s, y_t1, y_s, X_t2, 10)
    print v_pred

    auc = roc_auc_score(y_t2, v_pred)
    print auc

    #auc = roc_auc_score(y_t, v_t)
    #print auc

if __name__ == '__main__':
    # init
    _transform_func_degree = no_transform
    _transform_func = no_transform

    if sys.argv[1] == 'epinions':
        f_source_graph = 'epinions/out.epinions'
    else:           # 'slashdotzoo'
        f_source_graph = 'slashdot-zoo/out.matrix'

    main(f_source_graph)
